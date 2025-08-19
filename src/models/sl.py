import logging

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, Callback
from keras.src.layers import Dense
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.constants import Price
from src.models.models import CustomModel
from src.trade.dp import DPTable
from src.trade.env import TradeEnv
from src.trade.trade import tf_action_to_order, tf_execute_trade, tf_state_price, tf_norm_linspace_interp

logger = logging.getLogger(__name__)

class TQDMProgressBar(Callback):
    """A custom Keras callback that displays a TQDM progress bar."""

    def __init__(self, leave: bool = False, **tqdm_kwargs):
        """Initialize the callback."""
        super().__init__()
        self.tqdm_kwargs = tqdm_kwargs
        self.leave = leave
        self.progbar: tqdm = None

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.progbar = tqdm(
            total=self.params['epochs'],
            leave=self.leave,
            **self.tqdm_kwargs
        )

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        logs = logs or {}
        self.progbar.set_postfix(logs)
        self.progbar.update(1)

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.progbar:
            self.progbar.close()

class SuboptimalityDataGenerator(Sequence):
    """Generates batches of data for training the DPSL model."""

    def __init__(self, window_prices: np.ndarray, window_features: np.ndarray, window_indices: np.ndarray, batch_size: int):
        super().__init__()
        assert batch_size > 0, "batch_size must be non-negative"
        self.window_prices = window_prices
        self.window_features = window_features
        self.window_indices = window_indices

        self.n_samples = len(self.window_indices)
        self.batch_size = batch_size

        self.on_epoch_end()

    def on_epoch_end(self):
        """Shuffles the input and output context at the end of each epoch."""
        np.random.shuffle(self.window_indices)

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return self.n_samples // self.batch_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Generates one batch of data."""

        # Get indices for this batch
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indices = self.window_indices[start:end]

        # Calculate input
        batch_features = self.window_features[batch_indices]
        batch_exposures = np.random.uniform(-1.0, 1.0, size=(self.batch_size, 1))
        X_batch = np.hstack((batch_features, batch_exposures))

        # Calculate output context
        decision_indices = np.maximum(0, batch_indices - 1).reshape(-1, 1)
        batch_indices = batch_indices.reshape(-1, 1)
        decision_prices = self.window_prices[decision_indices, [Price.CLOSE_BID, Price.CLOSE_ASK]]
        exec_prices = self.window_prices[batch_indices, [Price.EXEC_BID, Price.EXEC_ASK]]
        close_prices = self.window_prices[batch_indices, [Price.CLOSE_BID, Price.CLOSE_ASK]]
        batch_indices_col = batch_indices.reshape(-1, 1)
        Y_batch = np.hstack([batch_exposures, decision_prices, exec_prices, close_prices, batch_indices_col])

        return X_batch, Y_batch

class DPSLModel(CustomModel):

    def __init__(self,
                 env: TradeEnv,
                 update_freq: int = 512,
                 lookback: int = 16_384,
                 n_actions: int = 15,
                 n_exposures: int = 15,
                 batch_size: int = 1024,
                 train_val_split = 0.8,
                 patience = 32):
        # Setup attributes
        self.env = env
        self.update_freq = update_freq
        self.lookback = lookback
        self.n_actions = n_actions
        self.n_exposures = n_exposures
        self.train_val_split = train_val_split
        self.patience = patience
        self.batch_size = batch_size

        # Setup updating
        self.model = self._build_model()
        self.dp_table = None
        self.last_update_t = self.env.t_start - self.update_freq

        # Setup indices:
        # For every timeframe we assign it to validation or training initially.
        # Then, we sort them such that we can easily find all the train and validation indices within our lookback window.
        # We provide this information to the
        self.train_indices, self.val_indices = train_test_split(np.arange(self.env.episode_len), test_size=self.train_val_split, shuffle=True)
        self.train_indices.sort()
        self.val_indices.sort()

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

    def _get_suboptimality_loss(self):

        value_table = tf.constant(self.dp_table.value_table, dtype=tf.float32)

        @tf.function
        def suboptimality_loss(y_true, y_pred):

            # Retrieve context
            target_exposure = y_pred[:, 0:1]
            prev_exposure = y_true[:, 0:1]
            decision_bid = y_true[:, 1:2]
            decision_ask = y_true[:, 2:3]
            exec_bid = y_true[:, 3:4]
            exec_ask = y_true[:, 4:5]
            close_bid = y_true[:, 5:6]
            close_ask = y_true[:, 6:7]
            window_indices = tf.cast(y_true[:, 7:8], dtype=tf.int32)

            # Norm equity cash and share calculations
            prev_cash = 1.0 - prev_exposure
            prev_val_price = tf.where(prev_exposure >= 0, decision_bid, decision_ask)
            prev_shares = prev_exposure / prev_val_price

            # Trade
            order_size = tf_action_to_order(prev_exposure, target_exposure, prev_cash, prev_shares, decision_bid, decision_ask, self.env.commission_pct)
            cash, shares = tf_execute_trade(prev_cash, prev_shares, order_size, exec_bid, exec_ask, self.env.commission_pct)
            _, _, next_exposure, log_equity = tf_state_price(cash, shares, close_bid, close_ask)

            # Calculate state-action (Q) value
            v_next = tf.gather(value_table, window_indices + 1, axis=0)
            v_next_interp = tf_norm_linspace_interp(next_exposure, v_next)
            q_value_model = log_equity + v_next_interp

            # Calculate state (V) value
            v_current = tf.gather(value_table, window_indices, axis=0)
            v_optimal = tf_norm_linspace_interp(prev_exposure, v_current)

            # Return suboptimality loss
            return tf.maximum(v_optimal - q_value_model, 0.0)

        return suboptimality_loss

    def _update_and_retrain(self):
        """Updates the DP table and retrains the model on the current lookback window."""

        # Calculate start and end
        start_idx = max(0, self.env.t - self.lookback)
        end_idx = self.env.t + 1 # exclusive
        logger.info(f"Retraining on data from index {start_idx} to {end_idx}")

        # Get window data
        window_prices = self.env.prices[start_idx:end_idx]
        window_features = self.env.features[start_idx:end_idx]

        val_start = np.searchsorted(self.val_indices, start_idx)
        val_end = np.searchsorted(self.val_indices, end_idx)
        window_val_indices = self.val_indices[val_start:val_end] - start_idx

        train_start = np.searchsorted(self.train_indices, start_idx)
        train_end = np.searchsorted(self.train_indices, end_idx)
        window_train_indices = self.train_indices[train_start:train_end] - start_idx

        # Calculate the DP table for the current lookback window
        self.dp_table = DPTable.compute(
            prices=window_prices,
            commission_pct=self.env.commission_pct,
            n_actions=self.n_actions,
            n_exposures=self.n_exposures
        )

        # Setup training generator
        train_generator = SuboptimalityDataGenerator(
            window_prices=window_prices,
            window_features=window_features,
            window_indices=window_train_indices,
            batch_size=self.batch_size,
        )

        val_generator = SuboptimalityDataGenerator(
            window_prices=window_prices,
            window_features=window_features,
            window_indices=window_val_indices,
            batch_size=self.batch_size
        )

        # Compile and train model
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )

        loss_fn = self._get_suboptimality_loss()
        self.model.compile(optimizer='adam', loss=loss_fn)
        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=256, # should be high, we should stop because of early_stopping
            callbacks=[early_stopping, TQDMProgressBar(desc="Training model")],
            verbose=0
        )

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Makes a prediction and triggers retraining with some update freq."""
        if self.env.t - self.last_update_t >= self.update_freq:
            self._update_and_retrain()
            self.last_update_t = self.env.t
        obs_tensor = tf.convert_to_tensor(np.atleast_2d(observation), dtype=tf.float32)
        prediction_tensor = self.model(obs_tensor, training=False)
        return prediction_tensor.numpy().flatten()