import logging

from RQ1.constants import RQ1_EXPERIMENTS_DIR
from RQ1.main import train_eval_analyze
from RQ1.parameters import ExperimentConfig, cud_colors, markers
from common.models.train_eval import combine_finals


def net_arch(shape: list):
    return dict(pi=shape, qf=shape)

if __name__ == "__main__":

    experiments = []
    for depth_name, depth, color, marker in zip(["shallow", "moderate", "deep", "very_deep"], [1, 2, 3, 4], cud_colors[:4], markers[:4]):
        for width_name, width, lookback, linestyle in zip(["narrow", "moderate", "wide"], [16, 32, 64], [2, 3, 4], ["-", "--", ":"]):
            experiments.append(ExperimentConfig(
                name = f"{depth_name}_{width_name}",
                net_arch = net_arch([width] * depth),
                lookback = lookback,
                line_marker = marker,
                line_color = color,
                line_style = linestyle,
            ))

    experiments.append(ExperimentConfig(
        name="bonus",
        net_arch=net_arch([128, 128, 128]),
        lookback=5,
    ))

    experiment_group = "network_sizes2"

    # print(*experiments, sep="\n")
    # for experiment in experiments:
    #     logging.info(f"Running experiment: {experiment}")
    #     train_eval_analyze(experiment, force=False, experiment_group=experiment_group)

    combine_finals(RQ1_EXPERIMENTS_DIR / experiment_group, {exp.name : exp.get_style() for exp in experiments}, ext=".svg")


