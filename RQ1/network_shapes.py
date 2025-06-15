import logging

from RQ1.constants import RQ1_EXPERIMENTS_DIR
from RQ1.main import train_eval_analyze
from RQ1.network_sizes import net_arch
from RQ1.parameters import ExperimentConfig, cud_colors, markers
from common.models.train_eval import combine_finals

if __name__ == "__main__":

    experiments = [
        ExperimentConfig(name="shape_flat", net_arch=net_arch([80, 80, 80]), lookback=3, line_color=cud_colors[0], line_marker="s"),
        ExperimentConfig(name="shape_funnel", net_arch=net_arch([120, 80, 40]), lookback=3, line_color=cud_colors[1], line_marker=">"),
        ExperimentConfig(name="shape_inv_funnel", net_arch=net_arch([40, 80, 120]), lookback=3, line_color=cud_colors[2], line_marker="<"),
        ExperimentConfig(name="shape_diamond", net_arch=net_arch([60, 120, 60]), lookback=3, line_color=cud_colors[3], line_marker="D"),
    ]

    experiment_group = "network_shape"

    print(*experiments, sep="\n")
    for experiment in experiments:
        logging.info(f"Running experiment: {experiment}")
        train_eval_analyze(experiment, force=False, experiment_group=experiment_group)

    combine_finals(RQ1_EXPERIMENTS_DIR / experiment_group, {exp.name : exp.get_style() for exp in experiments}, ext=".svg")
