import json
import logging
from asyncio import FIRST_EXCEPTION
from concurrent.futures import ProcessPoolExecutor, wait
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
import torch
import numpy as np
from tqdm import tqdm

from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.models.analysis import analyse_finals, analyse_individual_run
from common.models.dummy_models import DUMMY_MODELS
from common.models.utils import (load_model_with_metadata, save_model_with_metadata)
from common.scripts import parallel_map, set_seed, safe_int, parallel_apply

from common.models.dummy_models import DummyModel

import matplotlib
matplotlib.use('Agg') 

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

    seen: set[Path] = set()
    if not force_eval:
        for model_zip in models_dir.glob("*.zip"):
            if (results_dir / model_zip.stem).exists():
                logging.info(f"{model_zip} has already been evaluated, skipping.")
                seen.add(model_zip)

    while True:
        models = sorted(models_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime)
        new_models = [model for model in models if model not in seen]
        if not new_models:
            break
        seen.update(new_models)
        parallel_apply(func, new_models, num_workers=num_workers)

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
    parallel_apply(func, result_files, num_workers=num_workers)

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
    <experiment_group>/<experiment_name>/[seed_<seed>/]results/<model_name>/<environment_name>/info.json
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
            assert len(rel) == 4 and rel[0] == "results", f"unexpected paths {info_path}"
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

def train_model_with_curiosity(
    model, curiosity_module, train_env, train_episodes, beta, model_save_path, callback=None
):
    import logging
    from collections import Counter
    from tqdm import tqdm
    import numpy as np
    import torch

    logging.info(f"Training with curiosity for {train_episodes} episodes...")

    env = DummyVecEnv([lambda: train_env])
    total_steps = train_episodes * train_env.episode_len
    obs = env.reset()

    # Initialize callbacks
    if callback:
        for cb in callback:
            cb.init_callback(model)
            cb.num_timesteps = 0

    curiosity_batch = []
    curiosity_batch_size = 32
    intrinsic_episode_rewards = []
    episode_idx = 0
    previous_action = None

    # Debug metrics
    action_counts = Counter()
    intrinsic_vals = []
    extrinsic_vals = []
    combined_vals = []
    curiosity_losses = []
    inverse_losses = []
    forward_losses = []
    log_interval = 5000

    # Much shorter forced exploration with gradual transition
    forced_exploration_steps = 2000  # Reduced significantly
    logging.info(f"⚠️  Epsilon-greedy exploration for first {forced_exploration_steps} steps, then continuing with lower epsilon.")

    # Track Q-values for debugging
    q_value_history = []

    for step in tqdm(range(total_steps)):
        state_tensor = torch.FloatTensor(obs).to(model.device)

        # Epsilon-greedy with gradual decay throughout training
        if step < forced_exploration_steps:
            # High exploration initially
            epsilon = 0.9 - (step / forced_exploration_steps) * 0.4  # 0.9 -> 0.5
        else:
            # Continue with lower but non-zero exploration
            remaining_steps = total_steps - forced_exploration_steps
            progress = (step - forced_exploration_steps) / remaining_steps
            epsilon = 0.5 * (1 - progress) + 0.1  # 0.5 -> 0.1

        # Action selection with epsilon-greedy
        if np.random.random() < epsilon:
            # Ensure truly uniform random selection
            action = np.random.randint(0, train_env.n_actions)
        else:
            # Model prediction
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)

        # Debug: Track Q-values periodically
        if step % 1000 == 0 and step >= model.learning_starts:
            with torch.no_grad():
                if hasattr(model, 'q_net'):
                    q_vals = model.q_net(state_tensor).cpu().numpy().flatten()
                    q_value_history.append((step, q_vals.copy()))
                    if len(q_value_history) > 10:
                        q_value_history.pop(0)

        mapped_action = float(train_env.actions[action])
        action_idx_tensor = torch.tensor([action], dtype=torch.long, device=model.device)
        action_one_hot = np.zeros(train_env.n_actions, dtype=np.float32)
        action_one_hot[action] = 1.0
        action_one_hot_tensor = torch.FloatTensor(action_one_hot).unsqueeze(0).to(model.device)

        # Step environment
        next_obs, extrinsic_reward, done, info = env.step([action])
        next_state_tensor = torch.FloatTensor(next_obs).to(model.device)

        # Curiosity reward with better scaling
        intrinsic_reward = curiosity_module.compute_intrinsic_reward(
            state_tensor, next_state_tensor, action_one_hot_tensor
        )[0]
        
        # More conservative clipping to avoid overwhelming extrinsic rewards
        clipped_intrinsic = np.clip(intrinsic_reward, 0.0, 0.5)

        # Less aggressive beta decay
        progress = min(step / (total_steps * 0.8), 1.0)  # Cap progress at 80% of training
        dynamic_beta = beta * (1.0 - 0.3 * progress)  # Much less decay
        combined_reward = extrinsic_reward + dynamic_beta * clipped_intrinsic

        # Remove entropy penalty - it might be causing bias
        # if previous_action is not None and action == previous_action:
        #     combined_reward -= entropy_penalty_weight
        previous_action = action

        # Save to buffer
        infos = [{"TimeLimit.truncated": done}]
        model.replay_buffer.add(obs, next_obs, np.array([action]), combined_reward, done, infos=infos)

        # More frequent model training - FIXED: Handle TrainFreq object and logger properly
        if step >= model.learning_starts:
            # Get the actual frequency value from TrainFreq object
            if hasattr(model.train_freq, 'frequency'):
                train_freq_val = model.train_freq.frequency
            else:
                # Fallback: try to get the value directly or use a default
                train_freq_val = getattr(model.train_freq, 'value', model.train_freq)
            
            if step % max(1, train_freq_val // 2) == 0:
                # Ensure logger is available before training
                if not hasattr(model, '_logger') or model._logger is None:
                    from stable_baselines3.common.logger import configure
                    model._logger = configure(folder=None, format_strings=[])
                
                # DQN.train() requires gradient_steps parameter
                model.train(gradient_steps=model.gradient_steps)

        # Curiosity batch training
        curiosity_batch.append((state_tensor, next_state_tensor, action_idx_tensor, action_one_hot_tensor))
        if len(curiosity_batch) >= curiosity_batch_size:
            states, next_states, action_idxs, action_one_hots = zip(*curiosity_batch)
            loss, inv_loss, fwd_loss = curiosity_module.update_batch(states, next_states, action_idxs, action_one_hots)
            curiosity_batch.clear()
            curiosity_losses.append(loss)
            inverse_losses.append(inv_loss)
            forward_losses.append(fwd_loss)

        # Log metrics
        action_counts[mapped_action] += 1
        intrinsic_vals.append(clipped_intrinsic)
        extrinsic_vals.append(extrinsic_reward)
        combined_vals.append(combined_reward)
        intrinsic_episode_rewards.append(clipped_intrinsic)

        if step > 0 and step % log_interval == 0:
            # More detailed logging
            total_actions = sum(action_counts.values())
            action_dist = {}
            for action_val in train_env.actions:  # Ensure we check all actions
                count = action_counts.get(action_val, 0)
                action_dist[action_val] = f"{count}/{total_actions} ({count/total_actions:.2%})"
            
            logging.info(f"[Step {step}] Epsilon: {epsilon:.3f}")
            logging.info(f"[Step {step}] Action distribution: {action_dist}")
            logging.info(f"[Step {step}] Avg intrinsic: {np.mean(intrinsic_vals):.5f}, "
                         f"extrinsic: {np.mean(extrinsic_vals):.5f}, "
                         f"combined: {np.mean(combined_vals):.5f}, "
                         f"beta: {dynamic_beta:.3f}")
            
            if curiosity_losses:
                logging.info(f"[Step {step}] Curiosity loss: {np.mean(curiosity_losses):.4f}, "
                             f"Inverse: {np.mean(inverse_losses):.4f}, Forward: {np.mean(forward_losses):.4f}")
            
            # Log Q-values if available
            if q_value_history and step >= model.learning_starts:
                latest_step, latest_q_vals = q_value_history[-1]
                logging.info(f"[Step {step}] Q-values [short={latest_q_vals[0]:.3f}, hold={latest_q_vals[1]:.3f}, long={latest_q_vals[2]:.3f}]")
            
            # Check if all actions are being explored
            missing_actions = set(train_env.actions) - set(action_counts.keys())
            if missing_actions:
                logging.warning(f"[Step {step}] Missing actions in recent window: {missing_actions}")
            
            action_counts.clear()
            intrinsic_vals.clear()
            extrinsic_vals.clear()
            combined_vals.clear()
            curiosity_losses.clear()
            inverse_losses.clear()
            forward_losses.clear()

        # End of episode
        obs = next_obs if not done else env.reset()
        if done:
            avg_intrinsic = np.mean(intrinsic_episode_rewards)
            logging.info(f"[Episode {episode_idx}] Avg intrinsic reward: {avg_intrinsic:.4f}")
            intrinsic_episode_rewards.clear()
            episode_idx += 1

        # Callbacks
        model.num_timesteps += 1
        if callback:
            for cb in callback:
                cb.num_timesteps = model.num_timesteps
                cb.locals = {
                    'obs': obs,
                    'actions': np.array([action]),
                    'rewards': combined_reward,
                    'dones': done,
                    'infos': infos
                }
                cb.on_step()

    save_model_with_metadata(model, model_save_path / "model_final.zip")
    torch.save(curiosity_module.state_dict(), model_save_path / "curiosity_module.pth")

    logging.info("Curiosity training complete.")

def evaluate_and_analyze_model(exp_dir, train_env, eval_env, eval_episodes=1):
    models_dir = exp_dir / "models"
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    logging.info("Evaluating model...")
    evaluate_models(models_dir, results_dir, {"train": train_env, "eval": eval_env}, eval_episodes)

    # Analyze
    logging.info("Analyzing results...")
    analyse_results(results_dir)
