import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(
    data_path: Path | None,
    x_path: Path | None,
    y_path: Path | None,
    target_column: str | None,
) -> pd.DataFrame:
    if data_path:
        if not target_column:
            raise ValueError("target_column is required when using data_path")
        data = pd.read_csv(data_path)
        if target_column not in data.columns:
            raise ValueError(f"target_column '{target_column}' not found in data")
        return data

    if not (x_path and y_path):
        raise ValueError("Provide either data_path with target_column or x_path and y_path")

    x_data = pd.read_csv(x_path)
    y_data = pd.read_csv(y_path)

    if y_data.shape[1] != 1:
        raise ValueError("y_path must contain exactly one column")

    target_name = target_column or y_data.columns[0]
    combined = x_data.copy()
    combined[target_name] = y_data.iloc[:, 0].values
    return combined


def run_automl(config: dict, dataset: pd.DataFrame) -> None:
    task = config["task"]
    target_column = config["target_column"]
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
    parser.add_argument("--target", help="Target column name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target:
        config["target_column"] = args.target
    dataset = load_dataset(args.data, args.x, args.y, config.get("target_column"))
    run_automl(config, dataset)


if __name__ == "__main__":
    main()
