import logging

from RQ1.main import train_eval_analyze
from RQ1.parameters import ExperimentConfig


def net_arch(shape: list):
    return dict(pi=shape, qf=shape)

if __name__ == "__main__":

    experiments = [
        ExperimentConfig(name="shape_flat", net_arch=net_arch([80, 80, 80]), lookback=3),
        ExperimentConfig(name="shape_funnel", net_arch=net_arch([120, 80, 40]), lookback=3),
        ExperimentConfig(name="shape_inv_funnel", net_arch=net_arch([40, 80, 120]), lookback=3),
        ExperimentConfig(name="shape_diamond", net_arch=net_arch([60, 120, 60]), lookback=3),
    ]

    for depth_name, depth in zip(["shallow", "moderate", "deep", "very_deep"], [1, 2, 3, 4]):
        for width_name, width in zip(["narrow", "moderate", "wide"], [16, 32, 64]):
            experiments.append(ExperimentConfig(
                name = f"{depth_name}_{width_name}",
                net_arch = net_arch([width] * depth),
                lookback = depth,
            ))

    experiments.append(ExperimentConfig(
        name="bonus",
        net_arch=net_arch([128, 128, 128]),
        lookback=5,
    ))

    print(*experiments, sep="\n")
    for experiment in experiments:
        logging.info(f"Running experiment: {experiment}")
        train_eval_analyze(experiment, force=False, experiment_group="network_shapes")
