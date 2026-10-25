# TP4 - Aprendizaje No Supervisado

## Ejercicio 1 - Kohonen

```shell
make ej1
```

## Ejercicio 2 - Hopfield

Ejecución por defecto:

```shell
make ej2
```

Ejecución de la tarea de reconocimiento de letras (2.2) pasando argumentos por CLI o archivo de configuración:

```shell
# Usando flags de CLI directamente
make ej2 ARGS="--task letters --letters J A L O --update-mode asynchronous"

# O usando un archivo de configuración TOML
make ej2 ARGS="-c configs/ej2/01_letras_distintas.toml"
```

También puede invocarse directamente con `uv`:

```shell
uv run ej2 --task letters --letters J A L O
```

