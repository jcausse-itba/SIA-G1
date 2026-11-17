# Estudio del VAE sobre código de fractales — ejercicio 2

Desde la carpeta `tp-5`, ejecutar:

```powershell
uv run python study/ej2/_main_.py
```

El script hace todo el recorrido, igual que `study/ej1` y `study/ej1b`:

1. **Dataset.** Si no existe `code_dataset/code/`, corre `code_dataset/generator.py`, que genera
   10 000 programas de fractales (semilla 2026). Cada programa se codifica con `CodeDataset`:
   índice de cada carácter en un vocabulario de 58 símbolos ÷ 57, con relleno hasta 1000.
2. **Búsqueda con Optuna** (TPE + MedianPruner, semilla 42) sobre el VAE.
3. **Entrenamiento final** de la mejor configuración, con más épocas.
4. **Evaluación**: compila con `code_dataset/Flex-Bison-Compiler` (vía `FractalCompilerService`,
   que en Windows usa WSL) las reconstrucciones y las muestras generadas, y escribe gráficos,
   galerías, métricas y el modelo.

## Cortar la ejecución

Igual que en los otros TPs, se puede terminar el proceso sin perder lo hecho:

| Tecla | Qué pasa |
|---|---|
| **Ctrl+C** (una vez) o **Shift+Q** | Termina el trial actual, no lanza más y sigue con el entrenamiento final y todos los gráficos. Durante el entrenamiento final, lo corta y evalúa el modelo como quedó. |
| **Ctrl+C** dos veces | Aborta en el momento. Se escriben los gráficos de Optuna con los trials ya terminados; no se hace el entrenamiento final. |

El estudio se guarda en `resultados/optuna.db`, así que se puede retomar con `--resume`.

## El dataset crece con la red

La red más chica (256-128, latente 8) entrena con 100 programas. El resto usa una cantidad
proporcional a sus parámetros, redondeada a múltiplos de 50 (mínimo 100, máximo 1000):

| Red (capas ocultas del encoder) | Parámetros (latente 16) | Programas (latente 8 / 16 / 32) |
|---|---|---|
| 256-128 | 585 480 | 100 / 100 / 100 |
| 512-256 | 1 301 000 | 200 / 200 / 250 |
| 512-512-256-256 | 1 957 896 | 350 / 350 / 350 |
| 512×4-256 | 2 876 936 | 500 / 500 / 500 |

El límite de tiempo por trial también escala con el dataset (`TIME_PER_TRIAL_BASE_SEC` por cada 100 programas).

## Espacio de búsqueda

| Parámetro | Valores |
|---|---|
| Red | las 4 de la tabla |
| Dimensión latente | 8, 16, 32 |
| Conexiones residuales | sí / no |
| Activación oculta | tanh, relu |
| β (`kl_weight`) | 1e-6 a 1e-2 (log) |
| Optimizador | adam, rmsprop, momentum, sgd |
| η | 1e-4 a 1e-2 (log) |
| Scheduler | constant, cosine_annealing, exponential_decay, linear_decay |
| `min_lr_ratio` / `decay_rate` | solo para el scheduler que lo usa |
| `momentum_beta` | solo con momentum |
| `beta1`, `beta2`, `epsilon` | solo con adam |
| Batch size | 8, 16, 32 |

Fijos: salida lineal, MSE + β·KL, latente lineal, 1500 épocas por trial (los schedulers dependen
del total de épocas) y 3000 en el entrenamiento final.

## Objetivo (se minimiza)

```
score = 0.40·(1 − programas reconstruidos que compilan)
      + 0.30·(1 − caracteres reconstruidos correctamente)
      + 0.15·(1 − muestras locales nuevas que compilan)     # z = μᵢ + 0.25·σᵢ·ε
      + 0.15·(1 − muestras de N(0, I) que compilan)
```

Por trial se compilan 60 reconstrucciones y 40 muestras de cada tipo. El pruner usa
`1 − caracteres OK`, medido cada 50 épocas (sin compilar, para que sea rápido).

Como los trials con redes más grandes entrenan con más programas, el score compara redes con
datasets distintos: una red grande que llega al mismo score resolvió un problema más difícil.

## Archivos generados (`study/ej2/resultados/`)

Gráficos de Optuna (los mismos que el TP3):

- `plot_optimization_history.html`
- `plot_param_importances.html`
- `plot_slice.html`
- `optuna_trials.csv`: todos los trials, con las métricas en las columnas `user_attrs_*`.

Gráficos y salidas del estudio:

- `exploracion_arquitecturas.html`: mejor trial por red y score contra parámetros (el tamaño del punto es la cantidad de programas).
- `arquitectura_propuesta.html`: diagrama de la red elegida.
- `entrenamiento_final.html`: pérdida total, MSE y β·KL por época.
- `espacio_latente_pca.html`: latente proyectado con PCA y coloreado por familia (el hover muestra el programa).
- `generacion.html`: copias y programas nuevos que compilan, según cómo se elige z.
- `dataset_muestras.html`: fractales del dataset por familia.
- `galeria_generacion.html`: programas nuevos (original contra generado), muestras de N(0, I) e interpolación.
- `programas_generados/`: código (`.txt`) e imagen (`.bmp`) de cada programa nuevo y de su original.
- `metricas_finales.json`, `configuracion_ganadora.toml` y `modelo_ganador.pkl`.

## Opciones

```powershell
uv run python study/ej2/_main_.py --trials 30            # menos trials
uv run python study/ej2/_main_.py --resume               # continúa resultados/optuna.db
uv run python study/ej2/_main_.py --skip-final           # solo búsqueda y gráficos de Optuna
uv run python study/ej2/_main_.py --final-epochs 5000
uv run python study/ej2/_main_.py --output-dir study/ej2/mi_resultado
```

El modelo guardado se puede abrir con el explorador del espacio latente (`latent_explorer.py`).
