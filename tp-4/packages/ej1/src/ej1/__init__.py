import json
import os

from .cli import CAMPAIGNS, parse_config


def main() -> None:
    cfg = parse_config()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "config_usada.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)

    # Imports diferidos: --help y la validación no cargan plotly/scipy.
    from .data import load_dataset
    from .pca import fit_pca

    task = cfg["task"]
    ds = load_dataset(cfg)

    if task in ("eda", "all"):
        from .eda import run_eda
        run_eda(ds, cfg)

    pca = None
    if task in ("pca", "all"):
        from .pca import run_pca
        pca = run_pca(ds, cfg)
    elif task in ("kohonen", "oja"):
        pca = fit_pca(ds)  # sin gráficos: sólo para comparar

    if task == "sweep" or task == "all":
        from .kohonen import run_sweep
        run_sweep(ds, cfg)

    assign = None
    if task in ("kohonen", "campaign", "all"):
        from .kohonen import run_kohonen
        if task == "campaign":  # sólo se necesita la asignación a regiones
            from .kohonen import assignment_table, fit_best_som, neuron_regions
            som, _ = fit_best_som(ds, cfg)
            assign = assignment_table(som, ds, neuron_regions(som, cfg["n_regions"]))
        else:
            _, assign = run_kohonen(ds, cfg, pca.scores if pca is not None else None)

    if task in ("oja", "all"):
        from .oja import run_oja
        assert pca is not None, "El resultado de PCA es necesario para Oja"
        run_oja(ds, cfg, pca)

    if task in ("campaign", "all"):
        from .campaign import run_campaign
        for camp in (CAMPAIGNS if task == "all" else (cfg["campaign"],)):
            run_campaign(ds, cfg, camp, assign)


if __name__ == "__main__":
    main()
