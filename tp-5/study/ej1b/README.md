# Estudio del denoising autoencoder — ejercicio 1b

Desde la carpeta `tp-5`, ejecutar:

```powershell
uv run python study/ej1b/_main_.py
```

El programa usa los 32 caracteres de `doc/font.h`, corrompe los píxeles de
entrada con ruido bit-flip y entrena para reconstruir las imágenes limpias.
Genera gráficos Plotly y métricas en `study/ej1b/resultados/`.

La búsqueda evalúa 81 combinaciones: tres arquitecturas de igual profundidad
(`35-32-16-2`, `35-48-24-2`, `35-64-32-2`), tres optimizadores, tres learning
rates y tres probabilidades de ruido para entrenamiento (`0.1`, `0.2`, `0.3`).
Cada prueba usa batch size 8, semilla 42 y hasta 500 épocas. Para seleccionar,
evalúa cada combinación sobre los mismos ejemplos
corrompidos: cinco corrupciones por patrón para cada nivel `0.05`, `0.1`,
`0.2`, `0.3` y `0.4`. El criterio prioriza menor error promedio de píxeles
reconstruidos en esos niveles, luego el peor nivel, la reconstrucción limpia,
BCE, cantidad de parámetros y épocas. La combinación ganadora se reentrena
hasta 1500 épocas para producir los informes.

Archivos principales:

- `arquitectura_propuesta.html`: diagrama de la arquitectura seleccionada.
- `exploracion_arquitecturas.html`: mejor combinación observada por arquitectura.
- `optimizadores_hiperparametros.html`: resultados por optimizador, learning rate y nivel de ruido de entrenamiento.
- `capacidad_denoising.html`: comparación del ruido de entrada con el error posterior a la reconstrucción.
- `ej1b/reconstructions.html` y `ej1b/training_and_pixel_errors.html`: reconstrucción limpia y entrenamiento.
- `ej1b/latent_space_2d.html` y `ej1b/generated_samples.html`: espacio latente y muestras decodificadas.
- `comparaciones.csv`: resultados de las 81 combinaciones; identifica la seleccionada.
- `perfil_ruido.csv`: métricas finales por nivel de ruido.
- `configuracion_ganadora.toml` y `modelo_ganador.pkl`: hiperparámetros seleccionados y modelo final entrenado.

Para cambiar el destino de los resultados:

```powershell
uv run python study/ej1b/_main_.py --output-dir study/ej1b/mi_resultado
```
