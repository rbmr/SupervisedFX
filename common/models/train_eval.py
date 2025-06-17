import json
import logging
from functools import partial
from multiprocessing import Lock, Manager, Process
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.models.analysis import analyse_finals, analyse_individual_run
from common.models.dummy_models import DUMMY_MODELS
from common.models.utils import (load_model_with_metadata, save_model_with_metadata)
from common.scripts import parallel_run, set_seed, safe_int

from common.models.dummy_models import DummyModel


def run_experiment_deprecated(train_env: ForexEnv,
                       validate_env: ForexEnv,
                       model: BaseAlgorithm,
                       base_folder_path: Path,
                       experiment_group_name: str,
                       experiment_name: str,
                       eval_env: ForexEnv | None = None,
                       train_episodes: int = 10,
                       eval_episodes: int = 1,
                       checkpoints: bool = False,
                       tensorboard_logging: bool = False,
                       seed = 42,
                       num_workers: int = 1
                       ) -> None:
    """
    Train the model on the training DataFrame, test it on the test DataFrame, and export the results.
    """

    set_seed(seed)

    # Set up folders
    experiment_path = base_folder_path / "experiments" / experiment_group_name / experiment_name
    results_path = experiment_path / "results"
    models_path = experiment_path / "models"

    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # model class
    if not isinstance(model, BaseAlgorithm):
        raise ValueError(f"Model must be an instance of BaseAlgorithm, got {type(model)}")

    # TRAINING THE MODEL

    # set tensorboard logging if enabled
    if tensorboard_logging:
        model.tensorboard_log = str(experiment_path / "tensorboard_logs")

    starting_episode = 0
    # in the models dir, find the latest model and load it if it exists (latest model is the newest by modification time)
    latest_model = max(models_path.glob("*.zip"), key=lambda p: p.stat().st_mtime, default=None)
    if latest_model and latest_model.is_file():
        logging.info(f"Loading latest model from {latest_model}")
        model = load_model_with_metadata(latest_model)
        starting_episode = int(latest_model.stem.split('_')[1])
        train_episodes -= starting_episode
        train_episodes = max(train_episodes, 0)  # Ensure non-negative episodes


    callbacks = []
    if checkpoints:
        save_on_episode_callback = SaveOnEpisodeEndCallback(models_dir=models_path)
        save_on_episode_callback.episode_num = starting_episode
        callbacks.append(save_on_episode_callback)

    train_model(model, train_env=train_env, train_episodes=train_episodes, callback=callbacks)

    if not checkpoints:
        save_path = models_path / f"model_{train_episodes}_episodes.zip"
        save_model_with_metadata(model, save_path)

    # EVALUATING THE MODEL

    evaluate_models(models_dir=models_path,
                    results_dir=results_path,
                    eval_envs={"train": train_env,
                               "validate": validate_env},
                    eval_episodes=eval_episodes,
                    num_workers=num_workers)

    # ANALYZING RESULTS

    analyse_results(
        results_dir=results_path,
        model_name_suffix=f"[{experiment_group_name}::{experiment_name}]",
        num_workers=num_workers
    )

    if eval_env:
        logging.info("Selecting best model based on custom score (Val_Perf - |Val_Perf - Train_Perf|)...")
        
        # 1. Collect performance data for all models from their info.json files
        model_performance = {}
        selection_metric = 'sharpe_ratio' # The metric used for performance (e.g., sharpe_ratio)

        for info_path in results_path.rglob('info.json'):
            env_name = info_path.parent.name
            if env_name in ['train', 'validate']:
                model_name = info_path.parent.parent.name
                model_performance.setdefault(model_name, {})
                try:
                    with open(info_path, 'r') as f:
                        data = json.load(f)
                    metric_value = data.get(selection_metric)
                    if metric_value is not None:
                        model_performance[model_name][env_name] = metric_value
                except (IOError, json.JSONDecodeError) as e:
                    logging.warning(f"Could not read or parse {info_path}: {e}")

        # 2. Calculate the custom score for each model and find the best one
        scored_models = []
        for model_name, perfs in model_performance.items():
            if 'train' in perfs and 'validate' in perfs:
                train_perf = perfs['train']
                val_perf = perfs['validate']
                score = val_perf - abs(val_perf - train_perf)
                
                model_zip_path = models_path / f"{model_name}.zip"
                if model_zip_path.is_file():
                    scored_models.append((score, model_zip_path, val_perf, train_perf))
                else:
                    logging.warning(f"Could not find model zip for scored model {model_name}, skipping.")

        # 3. If a best model is found, evaluate it on the final evaluation environment
        if not scored_models:
            logging.error("No models with both train and validate results found. Cannot perform final evaluation.")
        else:
            scored_models.sort(key=lambda x: x[0], reverse=True)
            best_score, best_model_path, best_val_perf, best_train_perf = scored_models[0]
            
            logging.info(f"Best model selected: {best_model_path.name}")
            logging.info(f" -> Score: {best_score:.4f} (Validation Sharpe: {best_val_perf:.4f}, Training Sharpe: {best_train_perf:.4f})")
            logging.info("Evaluating best model on the final evaluation environment...")

            # Evaluate the best model on the eval_env
            evaluate_model(
                model_zip=best_model_path,
                results_dir=results_path,
                eval_envs={"evaluation": eval_env}, # Saves results in a dedicated 'evaluation' folder
                eval_episodes=eval_episodes,
                progress_bar=True
            )

            # 4. Analyze just the new evaluation result to create its specific info.json and plots
            logging.info("Analyzing final evaluation result...")
            eval_data_path = results_path / best_model_path.stem / "evaluation" / "data.csv"
            if eval_data_path.is_file():
                analyze_result(
                    data_csv=eval_data_path,
                    model_name_suffix=f"[{experiment_group_name}::{experiment_name}]"
                )
            else:
                logging.error(f"Final evaluation did not produce a data file at {eval_data_path}")


def train_model(model: BaseAlgorithm,
                train_env: ForexEnv,
                train_episodes: int | float = 1,
                callback: list[BaseCallback] | None = None
                ) -> None:
    """
    Trains a model on a ForexEnv for a given number of episodes.
    """

    logging.info(f"Training {model.__class__.__name__} model for {train_episodes} episodes...")

    train_dummy_env = DummyVecEnv([lambda: train_env])
    model.set_env(train_dummy_env)
    total_timesteps = int(train_env.episode_len * train_episodes)

    if total_timesteps > 0:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1, progress_bar=True)
    else:
        logging.warning("Total timesteps is 0, skipping training.")

    logging.info("Training complete.")

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

def evaluate_model(model_zip: Path,
                   results_dir: Path,
                   eval_envs: dict[str, ForexEnv],
                   eval_episodes: int = 1,
                   progress_bar: bool = False) -> None:
    """
    Evaluates a model on a set of ForexEnvs for a given number of episodes.
    Saves results in results_dir.
    """
    if not model_zip.is_file():
        raise ValueError("model_zip doesnt exist.")
    if not model_zip.suffix == ".zip":
        raise ValueError("model_zip is not a zip file.")

    model_name = model_zip.stem
    model = load_model_with_metadata(model_zip)
    model_results_dir = results_dir / model_name

    for eval_env_name, eval_env in eval_envs.items():
        env_results_dir = model_results_dir / eval_env_name
        env_results_file = env_results_dir / "data.csv"
        eval_episode_length = eval_env.episode_len

        logging.info(f"Running model ({model_name}) on environment ({eval_env_name}) for {eval_episodes} episodes...")

        run_model(model=model,
                  env=eval_env,
                  data_path=env_results_file,
                  total_steps=eval_episodes * eval_episode_length,
                  deterministic=True,
                  progress_bar=progress_bar)

class ModelQueue:
    def __init__(self, models_dir: Path, seen, lock: Lock):
        if not models_dir.is_dir():
            raise ValueError(f"{models_dir} is not a directory")
        self.models_dir = models_dir
        self.seen = seen # Shared list
        self.lock = lock # Shared lock

    def get(self) -> Path | None:
        with self.lock:
            models = sorted(self.models_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
            for model_zip in models:
                if not model_zip.is_file():
                    continue
                if model_zip in self.seen:
                    continue
                self.seen.append(model_zip)
                return model_zip
        return None

def evaluate_worker(queue: ModelQueue, func: Callable):
    while True:
        model_path = queue.get()
        if model_path is None:
            break
        func(model_path)

def evaluate_models(models_dir: Path,
                    results_dir: Path,
                    eval_envs: dict[str, ForexEnv],
                    eval_episodes: int = 1,
                    force_eval: bool = False,
                    num_workers: int = 4) -> None:
    """
    Evaluates each model in a directory on a set of ForexEnvs for a given number of episodes.
    Saves results in results_dir.
    """
    logging.info("Starting evaluation...")

    progress_bar = num_workers == 1
    func = partial(
        evaluate_model,
        results_dir=results_dir,
        eval_envs=eval_envs,
        eval_episodes=eval_episodes,
        progress_bar=progress_bar,
    )
    manager = Manager()
    shared_lock = manager.Lock()
    shared_seen = manager.list()
    queue = ModelQueue(models_dir, shared_seen, shared_lock)

    if not force_eval:
        results_dir = models_dir.parent / "results"
        for model_zip in models_dir.glob("*.zip"):
            if (results_dir / model_zip.stem).exists():
                shared_seen.append(model_zip)

    workers = []
    for i in range(num_workers):
        p = Process(target=evaluate_worker, args=(queue, func), name=f"Worker-{i+1}" )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    logging.info("Finished evaluation.")

def analyze_result(data_csv: Path, model_name_suffix: str = ""):
    """
    Extracts the environment name from a data.csv file.
    """
    analyse_individual_run(data_csv, f"{data_csv.parent.parent.name}{model_name_suffix}")

def extract_n(p: Path) -> Optional[int]:
    """
    Extracts n in the following structure:
    .../results/model_<n>_<unit>/<env_name>/file
    Returns None if it couldn't be extracted.
    """
    return safe_int(p.parent.parent.name.split("_")[1])

def extract_key(p) -> tuple[bool, int]:
    """
    Used to sort by key, sorts by n if possible else puts it at the end.
    """
    n = extract_n(p)
    return n is None, n

def analyse_results(results_dir: Path, model_name_suffix: str = "", num_workers = 4) -> None:
    """
    Searches a directory for data.csv files, and performs analysis.
    Expected directory structure: /{model_name}/{env_name}/data.csv
    """
    logging.info("Analysing results...")

    # Validate input
    if not results_dir.exists():
        raise ValueError(f"Directory {results_dir} does not exist.")
    if not results_dir.is_dir():
        raise ValueError(f"{results_dir} is not a directory.")

    # Evaluate runs
    func = partial(analyze_result, model_name_suffix=model_name_suffix)
    result_files = list(results_dir.rglob("data.csv"))
    result_files.sort(key=extract_key) # Old to new
    parallel_run(func, result_files, num_workers=num_workers)

    # Aggregate environment results
    logging.info("Aggregating analysis results...")
    metrics = {}
    for result_file in result_files:
        env_dir = result_file.parent
        env_name = env_dir.name
        if env_name not in metrics:
            metrics[env_name] = []
        info_file = env_dir / "info.json"
        with open(info_file, "r") as f:
            info = json.load(f)
        metrics[env_name].append(info)

    logging.info("Analyzing aggregates...")
    for env_name, metrics in metrics.items():
        analyse_finals(metrics, output_dir=results_dir / f"final_{env_name}", env_name=env_name)

    logging.info("Analysis complete.")

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



def combine_finals(experiment_group: Path, style_map: Optional[dict[str, dict[str, str]]] = None, ext: str = ".png"):
    """
    Combines the results of an experiment group.
    - Generates combined plots for each metric and environment with meaningful x-axes.
    - Generates a graphical summary table for each individual model run.
    """
    if style_map is None:
        style_map = {}

    # Data capture remains the same, as the model_key already contains the numerical value
    results: Dict[str, Dict[str, Dict[str, Dict[Optional[int], List[Tuple[Tuple[bool, int], float, str]]]]]] = {}
    experiment_names = set()
    for exp_dir in experiment_group.iterdir():
        if not exp_dir.is_dir(): continue
        exp_name = exp_dir.name
        experiment_names.add(exp_name)
        for info_path in exp_dir.rglob('info.json'):
            parts = info_path.relative_to(exp_dir).parts
            if parts[0].startswith("seed_"):
                seed = safe_int(parts[0].lstrip("seed_"))
                rel = parts[1:]
            else:
                seed = None
                rel = parts
            assert len(rel) == 4 and rel[0] == "results"
            env_name = rel[-2]
            model_name = info_path.parent.parent.name
            model_key = extract_key(info_path) # model_key is the (bool, int) tuple
            with open(info_path, mode="r") as f: data = json.load(f)
            
            env_dict = results.setdefault(env_name, {})
            for metric, val in data.items():
                metric_dict = env_dict.setdefault(metric, {})
                exp_dict = metric_dict.setdefault(exp_name, {})
                seed_list = exp_dict.setdefault(seed, [])
                seed_list.append((model_key, val, model_name))

    # Color generation logic remains the same
    sorted_experiment_names = sorted(list(experiment_names))
    experiments_to_color = [
        exp_name for exp_name in sorted_experiment_names
        if exp_name not in style_map or 'color' not in style_map.get(exp_name, {})
    ]
    if experiments_to_color:
        # ... (color generation logic is unchanged) ...
        num_experiments_to_color = len(experiments_to_color)
        if num_experiments_to_color <= 10: cmap = plt.cm.get_cmap('tab10')
        elif num_experiments_to_color <= 20: cmap = plt.cm.get_cmap('tab20')
        else:
            logging.warning("More than 20 experiments detected. Colors may not be perfectly distinct.")
            cmap = plt.cm.hsv
        colors = [cmap(i % cmap.N if cmap.N > 0 else 0) for i in range(num_experiments_to_color)]
        for i, exp_name in enumerate(experiments_to_color):
            if exp_name not in style_map: style_map[exp_name] = {}
            style_map[exp_name]['color'] = colors[i]

    # --- CHANGE 1: Capture X-Values During Data Processing ---
    # We now create model_x_values alongside the other dictionaries
    processed: Dict[str, Dict[str, List[float]]] = {}
    model_identifiers: Dict[str, Dict[str, List[str]]] = {}
    model_x_values: Dict[str, Dict[str, List[int]]] = {} # New dictionary to hold x-axis values

    for env, metrics in results.items():
        for metric, exp_map in metrics.items():
            for exp_name, seed_map in exp_map.items():
                if exp_name not in model_identifiers.setdefault(env, {}):
                    first_seed = next(iter(seed_map.values()), [])
                    first_seed.sort(key=lambda x: x[0])
                    # Store the string names for table filenames
                    model_identifiers[env][exp_name] = [name for _, _, name in first_seed]
                    # Store the numerical keys for the plot's x-axis
                    # The key is x[0], which is (bool, int). The value is the int part, x[0][1].
                    model_x_values.setdefault(env, {})[exp_name] = [key[1] for key, _, _ in first_seed]

                sorted_vals_per_seed: List[List[float]] = []
                for entries in seed_map.values():
                    entries.sort(key=lambda x: x[0])
                    vals = [v for _, v, _ in entries]
                    sorted_vals_per_seed.append(vals)

                if not any(sorted_vals_per_seed): continue
                lengths = {len(v) for v in sorted_vals_per_seed}
                if len(lengths) > 1:
                    logging.warning(f"Unequal lengths for {env}/{metric}/{exp_name}: {lengths}. Skipping.")
                    continue

                arr = np.array(sorted_vals_per_seed)
                out_env = processed.setdefault(env, {})
                out_env[f"{metric}.mean.{exp_name}"] = arr.mean(axis=0).tolist()
                out_env[f"{metric}.std.{exp_name}"] = arr.std(axis=0).tolist()

    # Output generation for tables and plots
    combined_dir = experiment_group / 'combined_finals'
    combined_dir.mkdir(exist_ok=True)

    for env, metrics_data in processed.items():
        # Table generation logic is unchanged
        base_metrics = sorted(list(set(k.split('.')[0] for k in metrics_data)))
        if model_identifiers.get(env):
            for exp_name in sorted(model_identifiers.get(env, {}).keys()):
                run_identifiers = model_identifiers[env][exp_name]
                output_dir_exp = combined_dir / env / exp_name
                output_dir_exp.mkdir(parents=True, exist_ok=True)
                for i, run_name in enumerate(run_identifiers):
                    # ... (table generation logic is unchanged) ...
                    table_data = []
                    for metric in base_metrics:
                        mean_val = metrics_data.get(f"{metric}.mean.{exp_name}", [])[i]
                        std_val = metrics_data.get(f"{metric}.std.{exp_name}", [])[i]
                        table_data.append([f'{mean_val:.4f}', f'{std_val:.4f}'])

                    fig, ax = plt.subplots(figsize=(4, 0.5 + len(base_metrics) * 0.3))
                    ax.axis('tight'); ax.axis('off')
                    table = plt.table(cellText=table_data, rowLabels=base_metrics, colLabels=['Mean', 'Std Dev'],
                                      loc='center', cellLoc='center')
                    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.4)
                    output_path = output_dir_exp / f"{run_name}_analysis_table{ext}"
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)
            logging.info(f"Generated analysis tables for environment '{env}'")

        # --- CHANGE 2 & 3: Modify Plotting Logic ---
        env_plot_dir = combined_dir / env
        for base in base_metrics:
            fig, ax = plt.subplots(figsize=(12, 6))

            for exp_name in sorted_experiment_names:
                mean_key = f"{base}.mean.{exp_name}"
                if mean_key not in metrics_data: continue
                std_key = f"{base}.std.{exp_name}"
                
                # Get the specific x-axis values for this experiment
                x_values = model_x_values.get(env, {}).get(exp_name)
                if not x_values: continue # Skip if no x-values were found

                mean = np.array(metrics_data[mean_key])
                std = np.array(metrics_data[std_key])
                style = style_map.get(exp_name, {})

                # Use the real x_values in all plotting commands
                if len(mean) > 1:
                    line = ax.plot(x_values, mean, label=exp_name, **style)
                    if std.max() > 0: ax.fill_between(x_values, mean - std, mean + std, color=line[0].get_color(), alpha=0.2, zorder=0)
                else:
                    plot_style = {**style, 'marker': style.get('marker', 'o'), 'linestyle': 'None'}
                    line = ax.plot(x_values, mean, label=exp_name, **plot_style)
                    if std.max() > 0: ax.errorbar(x_values, mean, yerr=std, fmt='none', color=line[0].get_color(), capsize=5)

            # Update the x-axis label to be more descriptive
            plt.title(f"{env} - {base}")
            plt.xlabel('Model Value (from name)') # CHANGE 3
            plt.ylabel(base)
            plt.legend(fontsize=10, loc='best', bbox_to_anchor=(1.0, 1.0))
            plt.tight_layout(pad=0.8)
            output = env_plot_dir / f"{base}{ext}"
            plt.savefig(output, bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)

    return processed