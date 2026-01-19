import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_targets(config: dict, cli_target: str | None) -> list[str]:
    if cli_target:
        return [cli_target]

    targets = config.get("target_columns")
    if targets:
        if isinstance(targets, str):
            return [target.strip() for target in targets.split(",") if target.strip()]
        if isinstance(targets, list):
            return [str(target).strip() for target in targets if str(target).strip()]
        raise ValueError("target_columns must be a list or comma-separated string")

    target = config.get("target_column")
    if target:
        return [str(target).strip()]

    raise ValueError("target_column or target_columns must be provided")


def resolve_data_paths(config: dict, args: argparse.Namespace) -> dict:
    def resolve_path(value: str | None) -> Path | None:
        return Path(value) if value else None

    return {
        "data_path": args.data or resolve_path(config.get("data_path")),
        "x_path": args.x or resolve_path(config.get("x_path")),
        "y_path": args.y or resolve_path(config.get("y_path")),
        "genotype_dir": args.genotype_dir or resolve_path(config.get("genotype_dir")),
        "phenotype_dir": args.phenotype_dir or resolve_path(config.get("phenotype_dir")),
    }


def load_dataset(
    data_path: Path | None,
    x_path: Path | None,
    y_path: Path | None,
    genotype_dir: Path | None,
    phenotype_dir: Path | None,
    target_columns: list[str] | None,
) -> pd.DataFrame:
    target_columns = target_columns or []
    if data_path:
        if not target_columns:
            raise ValueError("target_column(s) are required when using data_path")
        data = pd.read_csv(data_path)
        missing_targets = [col for col in target_columns if col not in data.columns]
        if missing_targets:
            raise ValueError(
                "target_column(s) not found in data: " + ", ".join(missing_targets)
            )
        return data

    if genotype_dir and phenotype_dir:
        if not target_columns:
            raise ValueError(
                "target_column(s) are required when using genotype/phenotype data"
            )
        return load_genotype_phenotype(genotype_dir, phenotype_dir, target_columns)

    if not (x_path and y_path):
        raise ValueError(
            "Provide data_path with target_column, x_path and y_path, or genotype/phenotype directories"
        )

    x_data = pd.read_csv(x_path)
    y_data = pd.read_csv(y_path)

    if y_data.shape[1] != 1:
        raise ValueError("y_path must contain exactly one column")
    if len(target_columns) > 1:
        raise ValueError("y_path only supports a single target column")

    target_name = target_columns[0] if target_columns else y_data.columns[0]
    combined = x_data.copy()
    combined[target_name] = y_data.iloc[:, 0].values
    return combined


def load_genotype_phenotype(
    genotype_dir: Path,
    phenotype_dir: Path,
    target_columns: list[str],
) -> pd.DataFrame:
    genotype_files = sorted(genotype_dir.glob("*.csv"))
    phenotype_files = sorted(phenotype_dir.glob("*.csv"))
    if not genotype_files:
        raise ValueError(f"No genotype CSV files found in {genotype_dir}")
    if not phenotype_files:
        raise ValueError(f"No phenotype CSV files found in {phenotype_dir}")

    genotype_df = pd.read_csv(genotype_files[0])
    phenotype_df = pd.read_csv(phenotype_files[0])

    meta_columns = ["SNP ID", "Chr", "Position (bp)"]
    missing_meta = [col for col in meta_columns if col not in genotype_df.columns]
    if missing_meta:
        raise ValueError(
            "Expected genotype columns missing: " + ", ".join(missing_meta)
        )

    if "Genotype" not in phenotype_df.columns:
        raise ValueError("Phenotype data must include a 'Genotype' column")
    missing_targets = [col for col in target_columns if col not in phenotype_df.columns]
    if missing_targets:
        raise ValueError(
            "target_column(s) not found in phenotype data: " + ", ".join(missing_targets)
        )

    snp_ids = genotype_df["SNP ID"].astype(str)
    genotype_values = genotype_df.drop(columns=meta_columns)
    genotype_transposed = genotype_values.T
    genotype_transposed.columns = snp_ids
    genotype_transposed.index.name = "Genotype"

    merged = phenotype_df.merge(
        genotype_transposed,
        left_on="Genotype",
        right_index=True,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No matching genotypes found between phenotype and genotype data")

    return merged


def run_automl(
    config: dict,
    dataset: pd.DataFrame,
    target_column: str,
    output_suffix: str | None = None,
) -> None:
    task = config["task"]
    cv_config = config.get("cross_validation", {})
    use_cv = cv_config.get("enabled", True)
    folds = cv_config.get("folds", 5)
    model_name = config.get("model_name")

    if task == "classification":
        from pycaret.classification import (
            compare_models,
            create_model,
            finalize_model,
            setup,
        )
    elif task == "regression":
        from pycaret.regression import (
            compare_models,
            create_model,
            finalize_model,
            setup,
        )
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    setup(
        data=dataset,
        target=target_column,
        session_id=config.get("session_id", 42),
        fold=folds if use_cv else 1,
        silent=True,
        verbose=False,
    )

    if model_name:
        model = create_model(model_name, fold=folds if use_cv else 1)
    else:
        model = compare_models(fold=folds if use_cv else 1)
    finalized = finalize_model(model)

    output_path = config.get("model_output_path")
    if output_path and output_suffix:
        output_path = f"{output_path}_{output_suffix}"
    if output_path:
        from pycaret.classification import save_model as save_cls
        from pycaret.regression import save_model as save_reg

        save_model_fn = save_cls if task == "classification" else save_reg
        save_model_fn(finalized, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyCaret AutoML")
    parser.add_argument("--config", required=True, type=Path, help="Path to config YAML")
    parser.add_argument("--data", type=Path, help="CSV containing features and target")
    parser.add_argument("--x", type=Path, help="CSV containing feature columns")
    parser.add_argument("--y", type=Path, help="CSV containing target column")
    parser.add_argument(
        "--genotype-dir",
        type=Path,
        help="Directory containing genotype CSV files",
    )
    parser.add_argument(
        "--phenotype-dir",
        type=Path,
        help="Directory containing phenotype CSV files",
    )
    parser.add_argument("--target", help="Target column name")
    args = parser.parse_args()

    config = load_config(args.config)
    targets = resolve_targets(config, args.target)
    data_paths = resolve_data_paths(config, args)
    dataset = load_dataset(
        data_paths["data_path"],
        data_paths["x_path"],
        data_paths["y_path"],
        data_paths["genotype_dir"],
        data_paths["phenotype_dir"],
        targets,
    )
    multiple_targets = len(targets) > 1
    for target in targets:
        run_automl(config, dataset, target, output_suffix=target if multiple_targets else None)


if __name__ == "__main__":
    main()
