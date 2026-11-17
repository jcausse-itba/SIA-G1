# Estudio del denoising autoencoder — ejercicio 1b

Desde la carpeta `tp-5`, ejecutar:

```powershell
uv run python study/ej1b/_main_.py
```

El programa usa los 32 caracteres de `doc/font.h`, corrompe los píxeles de
entrada con ruido bit-flip y entrena para reconstruir las imágenes limpias.
Genera gráficos Plotly y métricas en `study/ej1b/resultados/`.

La búsqueda evalúa 243 combinaciones: tres arquitecturas de igual profundidad
(`35-32-16-2`, `35-48-24-2`, `35-64-32-2`), tres optimizadores, tres learning
rates, tres probabilidades de ruido para entrenamiento (`0.1`, `0.2`, `0.3`)
y tres batch sizes (`4`, `8`, `16`). Cada prueba usa semilla 42 y hasta 500
épocas. Para seleccionar, evalúa cada combinación sobre los mismos ejemplos
corrompidos: cinco corrupciones por patrón para cada nivel `0.05`, `0.1`,
`0.2`, `0.3` y `0.4`. El criterio prioriza menor error promedio de píxeles
reconstruidos en esos niveles, junto con la cantidad de patrones dentro de la
tolerancia (hasta 1 píxel incorrecto). El umbral de binarización se barre de
`0.30` a `0.70` en pasos de `0.05`; no requiere reentrenar. La combinación
ganadora se reentrena hasta 1500 épocas.

Después, con arquitectura, optimizador, learning rate, ruido de entrenamiento
y batch size fijos, compara los schedulers durante hasta 1500 épocas:
`constant` (una ejecución), `cosine_annealing` y `linear_decay` con
`min_lr_ratio` en `0.01`, `0.02`, `0.05`, `0.10`, `0.20`, y
`exponential_decay` con `decay_rate` en esos mismos valores. Elige la
configuración priorizando letras limpias dentro de tolerancia y luego la
robustez en los niveles ruidosos.

Archivos principales:

- `arquitectura_propuesta.html`: diagrama de la arquitectura seleccionada.
- `exploracion_arquitecturas.html`: mejor combinación observada por arquitectura.
- `optimizadores_hiperparametros.html`: resultados por optimizador, learning rate, nivel de ruido de entrenamiento y batch size.
- `lr_scheduler_comparison.html` y `lr_scheduler_curves.html`: desempeño medido y curvas de learning rate de cada scheduler.
- `lr_schedulers.csv`: métricas de los schedulers; identifica el seleccionado.
- `threshold_comparison.html` y `threshold_comparison.csv`: métricas de cada umbral probado.
- `capacidad_denoising.html`: comparación del ruido de entrada con el error posterior a la reconstrucción.
- `efecto_ruido_caracteres.html`: grilla de 3 filas (original, ruidoso y reconstruido), niveles de ruido en columnas y selector de carácter.
- `ej1b/reconstructions.html` y `ej1b/training_and_pixel_errors.html`: reconstrucción limpia y entrenamiento.
- `ej1b/latent_space_2d.html` y `ej1b/generated_samples.html`: espacio latente y muestras decodificadas.
- `comparaciones.csv`: resultados de las 243 combinaciones iniciales.
- `perfil_ruido.csv`: métricas finales por nivel de ruido.
- `configuracion_ganadora.toml` y `modelo_ganador.pkl`: hiperparámetros seleccionados y modelo final entrenado.

Para cambiar el destino de los resultados:

```powershell
uv run python study/ej1b/_main_.py --output-dir study/ej1b/mi_resultado
```
