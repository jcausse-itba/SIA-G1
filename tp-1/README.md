# TP 1: Métodos de Búsqueda

## Tecnologías y Entorno

- **Lenguaje**: Python `>=3.14`
- **Gestor de Paquetes y Entorno**: [`uv`](https://github.com/astral-sh/uv)

- **Librerías Utilizadas**:
    - UI: `tkinter` (parte de la librería estándar de Python)
    - Generación de GIFs: `pillow` (instalada vía `uv`)
    - Plotting: `pandas`, `numpy` para el parseo de información y `plotly` para la creacion de graficos interactivos
    - Calculo de LAPJV: `scipy`

## Instalación y Ejecución

Este proyecto utiliza `uv` para la gestión determinista de dependencias y entornos virtuales.

1. Instalar `uv` (si no lo tiene en su sistema). Para ver instrucciones de instalación para su Sistema Operativo, haga [click aquí](https://docs.astral.sh/uv/getting-started/installation/).

2. Sincronizar Dependencias

    ```bash
    uv sync
    ```

3. Ejecución
    ```bash
    uv run sokoban-gui
    ```

## Comandos CLI

### Resolución de nivel

```sh
python.exe -m sokoban
```

> [!NOTE] OPCIONES
>
> Especificar el PATH Path al archivo .level
> `-p, --path PATH`
>
> Especificar el algoritmo de busqueda a usar (default: `astar`):
> `-a, --algorithm {dfs,bfs,astar,greedy}`
>
> Heuristica a usar (default: `none`)
> `--heuristic {min_goal_distance, unique_min_goal_distance, player_distance, hungarian}`
>
> Flag opcional para generar un GIF de la solucion, especificando el path al archivo .gif
> `-g, --gif GIF`

### Benchmark de coleccion de niveles

> [!NOTE] OPCIONES
>
> Especificar el directorio que contiene los archivos .level
> `-d, --dir DIR`
>
> Especificar el path del .csv retornado (default: `benchmark_results.csv`)
> `-o, --output OUTPUT`
>
> Buscar los archivos .level de manera recursiva (default: `False`)
> `-r, --recursive`
>
> Especificar el patrón para encontrar los niveles (default: `*.level`)
> `--pattern PATTERN`

### Mediciones de las benchmark de colección de niveles

> [!NOTE] OPCIONES
>
> Especificar el path al .csv
> `-c, --csv CSV`
>
> Especificar el directorio donde crear los graficos (default: `plots_output`)
> `-o, --outdir OUTDIR`
