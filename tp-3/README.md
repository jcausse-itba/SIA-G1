# TP 3: Perceptrón Simple y Multicapa

## Ejecutar usando un archivo de configuración

uv run python -m tp_3 -c config/ej1.toml

## Sobrescribir parámetros del archivo de configuración desde CLI

uv run python -m tp_3 -c config/ej1.toml --learning-rate 0.05 --max-epochs 2000

## Ejecutar pasando únicamente flags de línea de comandos

uv run python -m tp_3 -d ./data/digits.csv -m simple_non_linear --activation tanh

## CLI Reference & Configuration Options

### General Options

| Flag             | Argument | Description                                                                      | Default |
| ---------------- | -------- | -------------------------------------------------------------------------------- | ------- |
| `-h`, `--help`   | —        | Show this help message and exit.                                                 | —       |
| `-c`, `--config` | `CONFIG` | Path to a JSON/TOML configuration file. CLI arguments override JSON/TOML values. | `None`  |

---

### Dataset Parameters

| Flag                   | Argument            | Description                                 | Default    |
| ---------------------- | ------------------- | ------------------------------------------- | ---------- |
| `-d`, `--dataset-path` | `DATASET_PATH`      | Path to the training dataset CSV.           | _Required_ |
| `--test-dataset-path`  | `TEST_DATASET_PATH` | Path to the testing dataset CSV (optional). | `None`     |

---

### Model Architecture

| Flag                   | Argument     | Options / Format                                    | Description                                  | Default      |
| ---------------------- | ------------ | --------------------------------------------------- | -------------------------------------------- | ------------ |
| `-m`, `--model-type`   | `MODEL_TYPE` | `simple_linear`, `simple_non_linear` , `multilayer` | Type of perceptron model.                    | `multilayer` |
| `-a`, `--architecture` | `ARCH ...`   | Space-separated integers (e.g. `2 3 1`)             | Layer structure for Multilayer Perceptron.   | `[2, 3, 1]`  |
| `--activation`         | `ACTIVATION` | `step`, `linear`, `sigmoid`, `tanh`, `relu`         | Activation function.                         | `sigmoid`    |
| `--beta`               | `BETA`       | Float                                               | Beta coefficient for non-linear activations. | `1.0`        |

---

## Training & Optimization Parameters

| Flag                     | Argument        | Options / Format          | Description                              | Default |
| ------------------------ | --------------- | ------------------------- | ---------------------------------------- | ------- |
| `-lr`, `--learning-rate` | `LEARNING_RATE` | Float                     | Learning rate ($\eta$).                  | `0.01`  |
| `--optimizer`            | `OPTIMIZER`     | `sgd`, `momentum`, `adam` | Optimization algorithm.                  | `sgd`   |
| `--momentum-beta`        | `MOMENTUM_BETA` | Float                     | Beta coefficient for Momentum optimizer. | `0.9`   |
| `--batch-size`           | `BATCH_SIZE`    | Integer                   | Batch size for training.                 | `32`    |

---

## Stopping Conditions

| Flag                | Argument          | Type    | Description                                      | Default  |
| ------------------- | ----------------- | ------- | ------------------------------------------------ | -------- |
| `--max-epochs`      | `MAX_EPOCHS`      | Integer | Maximum number of epochs to train.               | `1000`   |
| `--target-error`    | `TARGET_ERROR`    | Float   | Target minimum error threshold to stop training. | `0.0001` |
| `--target-accuracy` | `TARGET_ACCURACY` | Float   | Target accuracy threshold to stop training.      | `0.98`   |

---

## Generalization Parameters

| Flag            | Argument      | Type                | Description                                   | Default |
| --------------- | ------------- | ------------------- | --------------------------------------------- | ------- |
| `--split-ratio` | `SPLIT_RATIO` | Float ($0.0 - 1.0$) | Train/Validation split ratio.                 | `0.8`   |
| `--k-folds`     | `K_FOLDS`     | Integer             | Number of folds for Cross-Validation.         | `5`     |
| `--threshold`   | `THRESHOLD`   | Float ($0.0 - 1.0$) | Decision threshold for binary classification. | `0.5`   |

---

## Model Persistence

| Flag                | Argument          | Description                                          | Default |
| ------------------- | ----------------- | ---------------------------------------------------- | ------- |
| `--save-model-path` | `SAVE_MODEL_PATH` | Path to save trained weights/model configuration.    | `None`  |
| `--load-model-path` | `LOAD_MODEL_PATH` | Path to load pre-trained weights to resume training. | `None`  |
