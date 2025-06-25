import logging
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from src.envs.forex_env import ForexEnv
from src.models.dummy_models import DUMMY_MODELS
from src.models.dummy_models import DummyModel

matplotlib.use('Agg')

def evaluate_dummy(dummy_model: DummyModel, name: str, results_dir: Path, eval_env: ForexEnv, eval_env_name: str) -> None:
    """
    Evaluates a dummy model on a ForexEnv and saves the results.
    """
    model_results_dir = results_dir / name
    env_results_dir = model_results_dir / eval_env_name
    env_results_file = env_results_dir / "data.csv"
    eval_episode_length = eval_env.episode_len

    logging.info(f"Running dummy model ({name}) on environment ({eval_env_name}) for 1 episode...")

    run_model(model=dummy_model,
              env=eval_env,
              data_path=env_results_file,
              total_steps=eval_episode_length,
              deterministic=True,
              progress_bar=True)

def evaluate_dummies(results_dir: Path, eval_envs: dict[str, ForexEnv]):

    for model_fn in DUMMY_MODELS:
        for eval_env_name, eval_env in eval_envs.items():

            model = model_fn(eval_env)
            model_name = model_fn.__name__

            logging.info(f"Running model ({model_name}) on environment ({eval_env_name}) for 1 episode...")

            evaluate_dummy(dummy_model=model,
                           name=model_name,
                           results_dir=results_dir,
                           eval_env=eval_env,
                           eval_env_name=eval_env_name)

# def evaluate_model(model_zip: Path,
#                    results_dir: Path,
#                    eval_envs: dict[str, ForexEnv],
#                    eval_episodes: int = 1,
#                    progress_bar: bool = False) -> None:
#     """
#     Evaluates a model on a set of ForexEnvs for a given number of episodes.
#     Saves results in results_dir.
#     """
#     if not model_zip.is_file():
#         raise ValueError("model_zip doesnt exist.")
#     if not model_zip.suffix == ".zip":
#         raise ValueError("model_zip is not a zip file.")
#
#     model_name = model_zip.stem
#     model = load_model_with_metadata(model_zip)
#     model_results_dir = results_dir / model_name
#
#     for eval_env_name, eval_env in eval_envs.items():
#         env_results_dir = model_results_dir / eval_env_name
#         env_results_file = env_results_dir / "data.csv"
#         eval_episode_length = eval_env.episode_len
#
#         logging.info(f"Running model ({model_name}) on environment ({eval_env_name}) for {eval_episodes} episodes...")
#
#         run_model(model=model,
#                   env=eval_env,
#                   data_path=env_results_file,
#                   total_steps=eval_episodes * eval_episode_length,
#                   deterministic=True,
#                   progress_bar=progress_bar)

# def evaluate_models(models_dir: Path,
#                     results_dir: Path,
#                     eval_envs: dict[str, ForexEnv],
#                     eval_episodes: int = 1,
#                     force_eval: bool = False,
#                     num_workers: int = 4) -> None:
#     """
#     Evaluates each model in a directory on a set of ForexEnvs for a given number of episodes.
#     Saves results in results_dir.
#     """
#     logging.info("Starting evaluation...")
#
#     progress_bar = num_workers == 1
#     func = partial(
#         evaluate_model,
#         results_dir=results_dir,
#         eval_envs=eval_envs,
#         eval_episodes=eval_episodes,
#         progress_bar=progress_bar,
#     )
#
#     seen: set[Path] = set()
#     if not force_eval:
#         for model_zip in models_dir.glob("*.zip"):
#             if (results_dir / model_zip.stem).exists():
#                 logging.info(f"{model_zip} has already been evaluated, skipping.")
#                 seen.add(model_zip)
#
#     while True:
#         models = sorted(models_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime)
#         new_models = [model for model in models if model not in seen]
#         if not new_models:
#             break
#         seen.update(new_models)
#         parallel_apply(func, new_models, num_workers=num_workers)
#
#     logging.info("Finished evaluation.")
#
# def analyze_result(data_csv: Path, model_name_suffix: str = ""):
#     """
#     Extracts the environment name from a data.csv file.
#     """
#     analyse_individual_run(data_csv, f"{data_csv.parent.parent.name}{model_name_suffix}")
#
# def extract_n(p: Path) -> Optional[int]:
#     """
#     Extracts n in the following structure:
#     .../results/model_<n>_<unit>/<env_name>/file
#     Returns None if it couldn't be extracted.
#     """
#     return safe_int(p.parent.parent.name.split("_")[1])
#
# def extract_key(p) -> tuple[bool, int]:
#     """
#     Used to sort by key, sorts by n if possible else puts it at the end.
#     """
#     n = extract_n(p)
#     return n is None, n
#
# def analyse_results(results_dir: Path, model_name_suffix: str = "", num_workers = 4) -> None:
#     """
#     Searches a directory for data.csv files, and performs analysis.
#     Expected directory structure: /{model_name}/{env_name}/data.csv
#     """
#     logging.info("Analysing results...")
#
#     # Validate input
#     if not results_dir.exists():
#         raise ValueError(f"Directory {results_dir} does not exist.")
#     if not results_dir.is_dir():
#         raise ValueError(f"{results_dir} is not a directory.")
#
#     # Evaluate runs
#     func = partial(analyze_result, model_name_suffix=model_name_suffix)
#     result_files = list(f for f in results_dir.rglob("data.csv") if not (f.parent / "info.json").exists())
#     result_files.sort(key=extract_key) # Old to new
#     parallel_apply(func, result_files, num_workers=num_workers)
#
#     # Aggregate environment results
#     logging.info("Aggregating analysis results...")
#     metrics = {}
#     for result_file in result_files:
#         env_dir = result_file.parent
#         env_name = env_dir.name
#         if env_name not in metrics:
#             metrics[env_name] = []
#         info_file = env_dir / "info.json"
#         with open(info_file, "r") as f:
#             info = json.load(f)
#         metrics[env_name].append(info)
#
#     logging.info("Analyzing aggregates...")
#     for env_name, metrics in metrics.items():
#         analyse_finals(metrics, output_dir=results_dir / f"final_{env_name}", env_name=env_name)
#
#     logging.info("Analysis complete.")

def run_model(model: BaseAlgorithm,
              env: ForexEnv,
              data_path: Path | None,
              total_steps: int,
              deterministic: bool,
              progress_bar: bool = True
              ) -> pd.DataFrame:
    """
    Run a model on a ForexEnv for a number of episodes.
    Results are saved to data_path.
    """
    # Validate input
    if data_path is not None:
        if data_path.suffix != ".csv":
            raise ValueError(f"{data_path} is not a CSV file")
        if total_steps <= 0:
            raise ValueError("Total steps must be greater than 0.")
        data_path.parent.mkdir(parents=True, exist_ok=True)

    # Function setup
    env = DummyVecEnv([lambda: env])
    steps = iter(tqdm(range(total_steps)) if progress_bar else range(total_steps))
    logs_df: None | pd.DataFrame = None

    # Start first episode
    obs = env.reset()
    episode_log: list[dict[str, Any]] = [{
        "step": 0,
        "action": None,
        "reward": None,
        "done": None,
    }]
    next(steps) # skip one value

    # Start the loop
    for step in steps:

        # Take action
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)
        episode_log.append({
            "step": step,
            "action": action[0] if isinstance(action[0], (int, float)) else action[0].item(),
            "reward": rewards[0],
            "done": dones[0],
        })

        # Check for end of episode
        if any(dones):

            # Save episode info
            info = infos[0] if infos else {}

            for obs_name, obs_df in info.items():
                if not isinstance(obs_df, pd.DataFrame):
                    logging.debug(f"Observation '{obs_name}' is not a DataFrame, skipping...")
                    continue
                obs_df.columns = [f"info.{obs_name}.{col}" for col in obs_df.columns]
                assert len(episode_log) == len(obs_df), f"len episode_log ({len(episode_log)}) != len {obs_name}_df ({len(obs_df)})"

            temp_df = pd.DataFrame(episode_log)
            temp_df = pd.concat(
                [temp_df] + [df for df in info.values() if isinstance(df, pd.DataFrame)],
                axis=1
            )
            logs_df = temp_df if logs_df is None else pd.concat([logs_df, temp_df], ignore_index=True, axis=0)

            # Start new episode
            obs = env.reset()
            episode_log = [{
                "step": 0,
                "action": None,
                "reward": None,
                "done": None,
            }]

    # Save collected logs to JSON file
    if data_path is not None:
        logs_df.to_csv(data_path, index=False)
    return logs_df


# def combine_finals(experiment_group: Path, style_map: Optional[dict[str, dict[str, str]]] = None, ext: str = ".png"):
#     """
#     Combines the results of an experiment group.
#     - Generates combined plots for each metric and environment with meaningful x-axes.
#     - Generates a graphical summary table for each individual model run.
#     <experiment_group>/<experiment_name>/[seed_<seed>/]results/<model_name>/<environment_name>/info.json
#     """
#     if style_map is None:
#         style_map = {}
#
#     # Data capture
#     results: Dict[str, Dict[str, Dict[str, Dict[Optional[int], List[Tuple[Tuple[bool, int], float, str]]]]]] = {}
#     experiment_names = set()
#     for exp_dir in experiment_group.iterdir():
#         if not exp_dir.is_dir(): continue
#         exp_name = exp_dir.name
#         experiment_names.add(exp_name)
#         for info_path in exp_dir.rglob('info.json'):
#             parts = info_path.relative_to(exp_dir).parts
#             if parts[0].startswith("seed_"):
#                 seed = safe_int(parts[0].lstrip("seed_"))
#                 rel = parts[1:]
#             else:
#                 seed = None
#                 rel = parts
#             assert len(rel) == 4 and rel[0] == "results", f"unexpected paths {info_path}"
#             env_name = rel[-2]
#             model_name = info_path.parent.parent.name
#             model_key = extract_key(info_path) # model_key is the (bool, int) tuple
#             with open(info_path, mode="r") as f: data = json.load(f)
#
#             env_dict = results.setdefault(env_name, {})
#             for metric, val in data.items():
#                 metric_dict = env_dict.setdefault(metric, {})
#                 exp_dict = metric_dict.setdefault(exp_name, {})
#                 seed_list = exp_dict.setdefault(seed, [])
#                 seed_list.append((model_key, val, model_name))
#
#     # Color generation logic
#     sorted_experiment_names = sorted(list(experiment_names))
#     experiments_to_color = [
#         exp_name for exp_name in sorted_experiment_names
#         if exp_name not in style_map or 'color' not in style_map.get(exp_name, {})
#     ]
#     if experiments_to_color:
#         num_experiments_to_color = len(experiments_to_color)
#         if num_experiments_to_color <= 10: cmap = plt.cm.get_cmap('tab10')
#         elif num_experiments_to_color <= 20: cmap = plt.cm.get_cmap('tab20')
#         else:
#             logging.warning("More than 20 experiments detected. Colors may not be perfectly distinct.")
#             cmap = plt.cm.hsv
#         colors = [cmap(i % cmap.N if cmap.N > 0 else 0) for i in range(num_experiments_to_color)]
#         for i, exp_name in enumerate(experiments_to_color):
#             if exp_name not in style_map: style_map[exp_name] = {}
#             style_map[exp_name]['color'] = colors[i]
#
#     # --- Capture X-Values During Data Processing ---
#     processed: Dict[str, Dict[str, List[float]]] = {}
#     model_identifiers: Dict[str, Dict[str, List[str]]] = {}
#     model_x_values: Dict[str, Dict[str, List[int]]] = {} # New dictionary to hold x-axis values
#
#     for env, metrics in results.items():
#         for metric, exp_map in metrics.items():
#             for exp_name, seed_map in exp_map.items():
#                 if exp_name not in model_identifiers.setdefault(env, {}):
#                     first_seed = next(iter(seed_map.values()), [])
#                     first_seed.sort(key=lambda x: x[0])
#                     # Store the string names for table filenames
#                     model_identifiers[env][exp_name] = [name for _, _, name in first_seed]
#                     # Store the numerical keys for the plot's x-axis
#                     # The key is x[0], which is (bool, int). The value is the int part, x[0][1].
#                     model_x_values.setdefault(env, {})[exp_name] = [key[1] for key, _, _ in first_seed]
#
#                 sorted_vals_per_seed: List[List[float]] = []
#                 for entries in seed_map.values():
#                     entries.sort(key=lambda x: x[0])
#                     vals = [v for _, v, _ in entries]
#                     sorted_vals_per_seed.append(vals)
#
#                 if not any(sorted_vals_per_seed): continue
#                 lengths = {len(v) for v in sorted_vals_per_seed}
#                 if len(lengths) > 1:
#                     logging.warning(f"Unequal lengths for {env}/{metric}/{exp_name}: {lengths}. Skipping.")
#                     continue
#
#                 arr = np.array(sorted_vals_per_seed)
#                 out_env = processed.setdefault(env, {})
#                 out_env[f"{metric}.mean.{exp_name}"] = arr.mean(axis=0).tolist()
#                 out_env[f"{metric}.std.{exp_name}"] = arr.std(axis=0).tolist()
#
#     # Output generation for tables and plots
#     combined_dir = experiment_group / 'combined_finals'
#     combined_dir.mkdir(exist_ok=True)
#
#     for env, metrics_data in processed.items():
#         # Table generation logic
#         base_metrics = sorted(list(set(k.split('.')[0] for k in metrics_data)))
#         if model_identifiers.get(env):
#             for exp_name in sorted(model_identifiers.get(env, {}).keys()):
#                 run_identifiers = model_identifiers[env][exp_name]
#                 output_dir_exp = combined_dir / env / exp_name
#                 output_dir_exp.mkdir(parents=True, exist_ok=True)
#                 for i, run_name in enumerate(run_identifiers):
#                     table_data = []
#                     for metric in base_metrics:
#                         mean_val = metrics_data.get(f"{metric}.mean.{exp_name}", [])[i]
#                         std_val = metrics_data.get(f"{metric}.std.{exp_name}", [])[i]
#                         table_data.append([f'{mean_val:.4f}', f'{std_val:.4f}'])
#
#                     fig, ax = plt.subplots(figsize=(4, 0.5 + len(base_metrics) * 0.3))
#                     ax.axis('tight'); ax.axis('off')
#                     table = plt.table(cellText=table_data, rowLabels=base_metrics, colLabels=['Mean', 'Std Dev'],
#                                       loc='center', cellLoc='center')
#                     table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.4)
#                     output_path = output_dir_exp / f"{run_name}_analysis_table{ext}"
#                     plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
#                     plt.close(fig)
#             logging.info(f"Generated analysis tables for environment '{env}'")
#
#         env_plot_dir = combined_dir / env
#         for base in base_metrics:
#             fig, ax = plt.subplots(figsize=(12, 6))
#
#             for exp_name in sorted_experiment_names:
#                 mean_key = f"{base}.mean.{exp_name}"
#                 if mean_key not in metrics_data: continue
#                 std_key = f"{base}.std.{exp_name}"
#
#                 # Get the specific x-axis values for this experiment
#                 x_values = model_x_values.get(env, {}).get(exp_name)
#                 if not x_values: continue # Skip if no x-values were found
#
#                 mean = np.array(metrics_data[mean_key])
#                 std = np.array(metrics_data[std_key])
#                 style = style_map.get(exp_name, {})
#
#                 # Use the real x_values in all plotting commands
#                 if len(mean) > 1:
#                     line = ax.plot(x_values, mean, label=exp_name, **style)
#                     if std.max() > 0: ax.fill_between(x_values, mean - std, mean + std, color=line[0].get_color(), alpha=0.2, zorder=0)
#                 else:
#                     plot_style = {**style, 'marker': style.get('marker', 'o'), 'linestyle': 'None'}
#                     line = ax.plot(x_values, mean, label=exp_name, **plot_style)
#                     if std.max() > 0: ax.errorbar(x_values, mean, yerr=std, fmt='none', color=line[0].get_color(), capsize=5)
#
#             # Update the x-axis label to be more descriptive
#             plt.title(f"{env} - {base}")
#             plt.xlabel('Model Value (from name)') # CHANGE 3
#             plt.ylabel(base)
#             plt.legend(fontsize=10, loc='best', bbox_to_anchor=(1.0, 1.0))
#             plt.tight_layout(pad=0.8)
#             output = env_plot_dir / f"{base}{ext}"
#             plt.savefig(output, bbox_inches='tight', pad_inches=0.2)
#             plt.close(fig)
#
#     return processed
