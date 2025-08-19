from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.src.layers import Dense
from stable_baselines3.common.base_class import BaseAlgorithm

from src.constants import Account
from src.trade.dp import DPTable
from src.trade.env import TradeEnv
from src.trade.trade import norm_linspace_interp


def get_dp_table_from_env(env: TradeEnv):
    return DPTable.get(env.prices, env.commission_pct, env.n_actions or 15, 15)

class CustomModel(ABC):

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        pass

class SB3ModelWrapper(CustomModel):

    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        actions, _ = self.model.predict(observation)
        return actions

class PerfectModel(CustomModel):

    def __init__(self, env: TradeEnv):
        self.env = env
        dp_table = get_dp_table_from_env(env)
        self.policy = dp_table.policy_table

    def predict(self, _: np.ndarray) -> np.ndarray:
        t = self.env.t
        prev_exposure = self.env.account[t, Account.CLOSE_EXPOSURE]
        return norm_linspace_interp(np.array([prev_exposure]), self.policy[t+1, :])

class RandomModel(CustomModel):
    def __init__(self, env: TradeEnv):
        self.env = env

    def predict(self, _: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()

class DPSLModel(CustomModel):

    def __init__(self,
                 env: TradeEnv,
                 update_freq: int = 256,
                 lookback: int = 16_384,
                 n_actions: int = 15,
                 n_exposures: int = 15
                 ):
        self.env = env
        self.update_freq = update_freq
        self.lookback = lookback
        self.n_actions = n_actions
        self.n_exposures = n_exposures
        self.model = self._build_model()
        self.dp_table = None
        self.last_update = self.env.t
        self._update_and_retrain()

    def _build_model(self):
        """Builds and compiles the Keras model."""
        input_dim = self.env.observation_space.shape[0]
        features_input = Input(shape=(input_dim,), name="features_input")
        x = Dense(48, activation='sigmoid')(features_input)
        x = Dense(48, activation='sigmoid')(x)
        x = Dense(48, activation='sigmoid')(x)
        target_exposure = Dense(1, activation='tanh', name="target_exposure")(x)
        model = Model(inputs=features_input, outputs=target_exposure)
        return model

    def _update_and_retrain(self):
        """Updates the DP table and retrains the model on the current lookback window."""
        #TODO: implement this

    def predict(self, observation: np.ndarray) -> np.ndarray:
        if self.env.t - self.last_update >= self.update_freq:
            self._update_and_retrain()
            self.last_update = self.env.t
        obs_tensor = tf.convert_to_tensor(np.atleast_2d(observation), dtype=tf.float32)
        prediction_tensor = self.model(obs_tensor, training=False)
        return prediction_tensor.numpy()


