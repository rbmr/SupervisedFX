
import os
import pandas as pd
from common.constants import Col
from common.scripts import combine_df, split_df, filter_df
from common.envs.forex_env import GeneralForexEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from common.feature.feature_engineer import FeatureEngineer
from common.feature.stepwise_feature_engineer import StepwiseFeatureEngineer
from common.scripts import run_model_on_vec_env, set_seed
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from typing import Any, Dict, List
import gymnasium as gym
from pathlib import Path


def train_test_export(train_df: pd.DataFrame, test_df: pd.DataFrame, environment: DummyVecEnv, model: gym.Env, base_folder_path: Path, experiment_group_name: str, experiment_name: str,
                        training_epochs: int = 10
                    ) -> None:
    """
    Train the model on the training DataFrame, test it on the test DataFrame, and export the results.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        The DataFrame to be used for training.
    test_df : pd.DataFrame
        The DataFrame to be used for testing.
    environment : DummyVecEnv
        The environment to be used for training and testing.
    model : gym.Env
        The model to be trained and tested.
    folder_path : Path
        The path where the results will be saved.
    experiment_group_name : str
        The name of the experiment group.
    experiment_name : str
        The name of the experiment.
    training_epochs : int, optional
        The number of epochs to train the model. Default is 10. The amount of times to train the model on the training DataFrame.
    """

    # check that train and test DataFrames have the same columns
    if set(train_df.columns) != set(test_df.columns):
        raise ValueError("Train and test DataFrames must have the same columns.")
    
    # Set seeds
    set_seed(42)

    # Set up folders
    experiment_path = base_folder_path / "experiments" / experiment_group_name / experiment_name
    results_path = experiment_path / "results"
    logs_path = experiment_path / "logs"
    folder_path = experiment_path / "models"

    folder_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)
    logger = configure(logs_path, ["stdout", "csv", "tensorboard"])

    
    # train the model (allow for saving several models)

    # TESTING ON TRAIN DATA FRAME
    # test the model(s) on the train data frame
    # save all relevant information in one big DataFrame
    # plot as many plots as possible with the results
    
    # TESTING ON TEST DATA FRAME
    # test the model(s) on the test data frame
    # save all relevant information in one big DataFrame
    # plot as many plots as possible with the results


    print("Done!")