import argparse


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser with all parameters and their defaults."""
    parser = argparse.ArgumentParser(
        description="TP2: Image Reconstruction with Genetic Algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to a JSON/TOML configuration file. CLI arguments override JSON/TOML values.")

    prob = parser.add_argument_group("Problem Parameters")
    prob.add_argument("-i", "--image-path", type=str, default="./input/target.png", help="Path to the target image")
    prob.add_argument("-o", "--output-path", type=str, default="./output/result.png", help="Path to save the output image")
    prob.add_argument("-t", "--num-triangles", type=int, default=50, help="Number of triangles per individual")

    ga = parser.add_argument_group("Genetic Algorithm Parameters")
    ga.add_argument("-N", "--pop-size", type=int, default=100, help="Population size (N)")
    ga.add_argument("-K", "--children-size", type=int, default=60, help="Number of children to generate per generation (K)")
    ga.add_argument("-pc", "--crossover-prob", type=float, default=0.8, help="Crossover probability")
    ga.add_argument("-pm", "--mutation-prob", type=float, default=0.05, help="Mutation probability")
    ga.add_argument("--survival-strategy", type=str, choices=["additive", "exclusive"], default="additive", help="Replacement strategy")
    ga.add_argument("-G", "--generation-gap", type=float, default=1.0, help="Generation gap (G)")

    methods = parser.add_argument_group("Genetic Strategies")
    methods.add_argument("--parent-selection", type=str, default="roulette",
                         choices=["elite", "roulette", "universal", "boltzmann", "det_tournament", "prob_tournament", "ranking"],
                         help="Parent selection method")
    methods.add_argument("--survival-selection", type=str, default="elite",
                         choices=["elite", "roulette", "universal", "boltzmann", "det_tournament", "prob_tournament", "ranking"],
                         help="Survival selection method")
    # TODO remove the ones we don't want
    methods.add_argument("--crossover", type=str, default="two_point",
                         choices=["one_point", "two_point", "uniform", "annular"], help="Crossover method")
    methods.add_argument("--mutation", type=str, default="multigene_uniform",
                         choices=["single_gene", "multigene_limited", "multigene_uniform", "complete"], help="Mutation method")

    params = parser.add_argument_group("Method Specific Parameters")
    params.add_argument("--tournament-m", type=int, default=3, help="Group size M for deterministic tournaments")
    params.add_argument("--tournament-threshold", type=float, default=0.75, help="Threshold U for probabilistic tournaments (0.5 to 1.0)")
    params.add_argument("--boltzmann-t0", type=float, default=100.0, help="Initial temperature T0 for Boltzmann")
    params.add_argument("--boltzmann-tc", type=float, default=10.0, help="Consolidated temperature Tc for Boltzmann")
    params.add_argument("--boltzmann-k", type=float, default=0.01, help="Cooling rate constant k")

    stop = parser.add_argument_group("Stopping Conditions")
    stop.add_argument("--stop-criterion", type=str, default="generations",
                      choices=["generations", "fitness", "structure", "content", "time"], help="Stopping criterion")
    stop.add_argument("--max-generations", type=int, default=1000, help="Maximum number of generations")
    stop.add_argument("--target-fitness", type=float, default=0.99, help="Target or minimum acceptable fitness")
    stop.add_argument("--stagnation-limit", type=int, default=50, help="Generation limit without changes for structure/content")
    stop.add_argument("--max-time-seconds", type=float, default=300.0, help="Maximum execution time in seconds")

    return parser