# Estudio del autoencoder — ejercicio 1a

Desde la carpeta `tp-5`, ejecutar:

```powershell
uv run python study/ej1/_main_.py
```

El comando usa `doc/font.h` (los 32 patrones de 7×5) y genera gráficos Plotly
interactivos y una tabla de resultados en `study/ej1/resultados/`. No requiere
instalar dependencias adicionales a las de `pyproject.toml`.

El estudio prueba las 27 combinaciones del producto cartesiano de tres
arquitecturas (`35-32-16-2`, `35-48-24-2`, `35-64-32-16-2`), tres optimizadores
(`sgd`, `momentum`, `adam`) y tres learning rates (`0.001`, `0.004`, `0.01`).
Usa batch size 8, semilla 42 y hasta 700 épocas por combinación; detiene cada
entrenamiento antes si alcanza el objetivo de máximo un píxel incorrecto.

La selección se hace con los resultados medidos, en este orden: menor máximo
de píxeles incorrectos por patrón, menor error promedio, menor BCE, menor
cantidad de parámetros y menos épocas. Después se reentrena esa combinación
ganadora hasta 1500 épocas para producir los gráficos finales. El proceso
termina con error si el modelo reentrenado no alcanza el límite de un píxel.
En los gráficos comparativos, cada barra muestra la mejor combinación
observada para esa arquitectura, optimizador o learning rate; la combinación
ganadora se resalta en verde. `comparaciones.csv` contiene las 27 pruebas y
marca la seleccionada.

Archivos principales:

- `arquitectura_propuesta.html`: diagrama de la arquitectura seleccionada.
- `exploracion_arquitecturas.html`: mejor combinación observada por arquitectura.
- `optimizadores_hiperparametros.html`: comparación de optimizadores y learning rates.
- `ej1/training_and_pixel_errors.html`: curva de entrenamiento y error por carácter.
- `ej1/reconstructions.html`: originales frente a reconstrucciones.
- `ej1/latent_space_2d.html`: representación de los caracteres en el espacio latente.
- `ej1/generated_samples.html` y `glifo_generado.html`: decodificaciones generadas desde el espacio latente.
- `comparaciones.csv`: métricas para incluir en el informe.
- `configuracion_ganadora.toml` y `modelo_ganador.pkl`: hiperparámetros seleccionados y modelo final entrenado.

También se puede cambiar el directorio de salida:

```powershell
uv run python study/ej1/_main_.py --output-dir study/ej1/mi_resultado
```
