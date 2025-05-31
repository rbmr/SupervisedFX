
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from common.envs.callbacks import SaveOnEpisodeEndCallback
from common.envs.forex_env import ForexEnv
from common.scripts import set_seed, flatten_dict


def train_test_analyze(train_env: ForexEnv,
                       eval_env: ForexEnv,
                       model: BaseAlgorithm,
                       base_folder_path: Path,
                       experiment_group_name: str,
                       experiment_name: str,
                       train_episodes: int = 10,
                       eval_episodes: int = 1,
                       checkpoints: bool = False,
                       deterministic: bool = True
                       ) -> None:
    """
    Train the model on the training DataFrame, test it on the test DataFrame, and export the results.
    
    Parameters
    ----------
    
    """
    # Set seeds
    set_seed(42)

    # Set up folders
    experiment_path = base_folder_path / "experiments" / experiment_group_name / experiment_name
    results_path = experiment_path / "results"
    logs_path = experiment_path / "logs"
    models_path = experiment_path / "models"

    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    # model class
    if not isinstance(model, BaseAlgorithm):
        raise ValueError(f"Model must be an instance of BaseAlgorithm, got {type(model)}")
    model_class = type(model)

    # set env
    train_dummy_env = DummyVecEnv([lambda: train_env])
    model.set_env(train_dummy_env)

    # total timesteps
    total_timesteps = train_env.max_episode_timesteps() * train_episodes

    # train the model (saving it every epoch)
    logging.info(f"Training model for {train_episodes} epochs...")
    callbacks = []
    if checkpoints:
        checkpoint_callback = SaveOnEpisodeEndCallback(save_path=str(models_path))
        callbacks.append(checkpoint_callback)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1, progress_bar=True)
    logging.info("Training complete.")

    # save the final model
    save_path = models_path / f"model_{total_timesteps}_steps.zip"
    model.save(save_path)
    logging.info(f"Model(s) saved to '{models_path}'.")
    
    
    # TESTING THE MODELS
    # 1. find all model zips in the models_path
    model_files = list(models_path.glob("*.zip"))
    if not model_files:
        logging.warning("No model files found in the models directory. Skipping evaluation.")
        return
    logging.info(f"Found {len(model_files)} model files in '{models_path}'.")

    # sort model files by modification time (oldest first)
    model_files.sort(key=lambda x: x.stat().st_mtime)

    # 2. Run Each Model on the train_env and eval_env
    for model_file in model_files:
        logging.info(f"Loading model from {model_file}...")
        model = model_class.load(model_file, env=train_dummy_env)
        logging.info(f"Model loaded from {model_file}.")

        this_model_path = results_path / model_file.stem
        train_data_path = this_model_path / "train"
        eval_data_path = this_model_path / "eval"
        train_results_full_file = train_data_path / "data"
        eval_results_full_file = eval_data_path / "data"
        train_data_path.mkdir(parents=True, exist_ok=True)

        train_episode_length = len(train_env.market_data_df)
        eval_episode_length = len(eval_env.market_data_df)

        run_model_on_vec_env(model, train_env, train_results_full_file, total_steps=eval_episodes * train_episode_length, deterministic=deterministic, progress_bar=True)

        run_model_on_vec_env(model, eval_env, eval_results_full_file, total_steps=eval_episodes * eval_episode_length, deterministic=deterministic, progress_bar=True)



    # ANALYSIS
    logging.info("Analyzing results...")
    model_train_metrics = []
    model_eval_metrics = []
    for model_file in model_files:
        model_name = model_file.stem
        this_model_path = results_path / model_name
        train_data_path = this_model_path / "train"
        eval_data_path = this_model_path / "eval"
        train_results_full_file = train_data_path / "data.csv"
        eval_results_full_file = eval_data_path / "data.csv"
        if not train_results_full_file.exists() or not eval_results_full_file.exists():
            logging.warning(f"Skipping analysis for {model_name} as one or both result files do not exist.")
            continue
        
        logging.info(f"Analyzing results for model: {model_name}")
        
        # Load train and eval results
        results_df = pd.read_csv(train_results_full_file)
        metrics = analyse_individual_run(results_df, train_data_path, name=model_name)
        model_train_metrics.append(metrics)

        results_df = pd.read_csv(eval_results_full_file)
        metrics = analyse_individual_run(results_df, eval_data_path, name=model_name)
        model_eval_metrics.append(metrics)

    analyse_finals(model_train_metrics, results_path, name="train_results")
    analyse_finals(model_eval_metrics, results_path, name="eval_results")
    
    logging.info("Analysis complete.")


    logging.info("Done!")

def run_model_on_vec_env(model: BaseAlgorithm, env: ForexEnv, data_path: Path, total_steps: int, deterministic: bool, progress_bar: bool = True) -> None:
    """
    Run a trained RL model on a vectorized environment for a number of episodes,
    log each step, and write all logs at the end. Optionally displays a progress bar.
    """

    if total_steps <= 0:
        logging.warning("Total steps must be greater than 0. No steps will be executed.")
        return

    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    collected_log_entries: List[Dict[str, Any]] = []

    env = DummyVecEnv([lambda: env])  # Ensure env is wrapped in DummyVecEnv
    
    obs = env.reset()
    
    pbar = None
    if progress_bar:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_steps, desc="Total Steps")
        except ImportError:
            print("Warning: tqdm is not installed. Progress bar will not be shown. "
                  "Install with: pip install tqdm")

    step_count = 1
    while step_count < total_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, rewards, dones, infos = env.step(action)

        for i in range(env.num_envs):
            log_entry: Dict[str, Any] = {
                "env_index": i,
                "step": step_count,
                "action": action[i].tolist(),
                "reward": rewards[i],
                "done": dones[i],
                "info": infos[i],
            }
            collected_log_entries.append(log_entry)

        if pbar:
            pbar.update(1)

        # If done, break out of the while loop
        if any(dones):
            if pbar:
                pbar.set_description("Episode completed")
            # Reset the environment
            obs = env.reset()

        obs = next_obs
        step_count += 1
    
    if pbar:
        pbar.close()

    # Save collected logs to JSON file
    # flatten each dictionary in collected_log_entries to be only one level deep. 
    flat_log_entries = [flatten_dict(entry) for entry in collected_log_entries]
    log_df = pd.DataFrame(flat_log_entries)
    log_df.to_csv(data_path.with_suffix('.csv'), index=False)






    
    