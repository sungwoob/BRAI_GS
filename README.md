# BRAI_GS

PyCaret AutoML runner with configurable input and cross-validation.

## Requirements

```bash
pip install pycaret pandas pyyaml
```

## Usage

### Configure

Edit `config.yaml` to set task, target column, model choice, cross-validation, and output path.

### Run with a single CSV (features + target)

```bash
python automl.py --config config.yaml --data path/to/data.csv --target target
```

### Run with separate X and Y CSVs

```bash
python automl.py --config config.yaml --x path/to/x.csv --y path/to/y.csv
```
