import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from tqdm.keras import TqdmCallback

from src.constants import MODELS_DIR, DP_CACHE_DIR, MarketDataCol
from src.data.data import ForexCandleData, Timeframe
from src.data.feature_engineer import FeatureEngineer
from src.data.stepwise_feature_engineer import StepwiseFeatureEngineer, calculate_current_exposure
from src.envs.dp import DPTable, get_dp_table, interp
from src.envs.forex_env import ForexEnv, ActionConfig, EnvConfig, EnvObs, DataConfig
from src.envs.trade import execute_trade_1equity
from src.models.analysis import analyse_individual_run
from src.models.dummy_models import DummyModel
from src.models.train_eval import run_model
from src.scripts import find_first_valid_row, contains_nan_or_inf

TRANSACTION_COST_PCT = 5 / 100_000
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SuboptimalityDataGenerator(Sequence):
    def __init__(self, features: np.ndarray, batch_size: int, global_indices: np.ndarray):
        super().__init__()
        assert batch_size > 0, "batch_size must be positive"
        self.features = features
        self.batch_size = batch_size
        self.n_samples = self.features.shape[0]
        self.global_indices = global_indices
        self.X = None
        self.Y = None
        self.on_epoch_end()

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], None]:
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        return self.X[start_idx:end_idx], self.Y[start_idx:end_idx]

    def on_epoch_end(self):
        p = np.random.permutation(self.n_samples)
        shuffled_features = self.features[p]
        next_indices = self.global_indices[p] + 1 # We are executing the actions at the start of the next timestep
        random_exposures = np.random.uniform(-1.0, 1.0, self.n_samples).reshape(-1, 1)
        self.X = np.column_stack((shuffled_features, random_exposures))
        self.Y = np.column_stack((next_indices, random_exposures))

def to_percentiles(tensor: tf.Tensor) -> tf.Tensor:
    """
    Converts each element in a tensor to its percentile rank (0.0 to 1.0).
    """
    flat_tensor = tf.reshape(tensor, [-1])
    sorted_unique_values = tf.sort(tf.unique(flat_tensor).y)
    ranks = tf.searchsorted(sorted_unique_values, flat_tensor, side='left')
    percentiles_flat = tf.cast(ranks, tf.float32) / tf.cast(tf.shape(sorted_unique_values)[0], tf.float32)
    return tf.reshape(percentiles_flat, tf.shape(tensor))


def dummy_loss(*_) -> float:
    return 0.0

def create_suboptimality_loss(dp_table: DPTable, market_data: np.ndarray, transaction_cost_pct: float):
    value = tf.constant(dp_table.value_table, dtype=tf.float32)
    q_min = tf.constant(dp_table.q_min_table, dtype=tf.float32)
    market_data = tf.constant(market_data, dtype=tf.float32)
    n_exposures = dp_table.n_exposures
    raw_importance = value - q_min
    percentile_importance = to_percentiles(raw_importance)

    def suboptimality_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        """The actual loss function, implementing percentile normalization."""
        current_indices = tf.cast(y_true[:, 0:1], dtype=tf.int32)
        current_exposures = y_true[:, 1:2]

        # This is the Q-value of the action the agent actually took
        next_equity, next_exposure = execute_trade_1equity(
            indices=current_indices,
            current_exposures=current_exposures,
            target_exposures=y_pred,
            market_data=market_data,
            transaction_cost_pct=transaction_cost_pct,
        )
        v_next_slices = tf.gather(value, tf.reshape(current_indices + 1, [-1]))
        v_next = interp(v_next_slices, next_exposure, n_exposures)
        reward = tf.math.log(tf.maximum(next_equity, 1e-9))
        q_predicted = reward + v_next

        # Get V_optimal, Q_min, and importance_percentile for the current state
        v_current_slices = tf.gather(value, tf.reshape(current_indices, [-1]))
        v_optimal = interp(v_current_slices, current_exposures, n_exposures)

        q_min_slices = tf.gather(q_min, tf.reshape(current_indices, [-1]))
        q_min_current = interp(q_min_slices, current_exposures, n_exposures)

        percentile_importance_slices = tf.gather(percentile_importance, tf.reshape(current_indices, [-1]))
        importance_percentile = interp(percentile_importance_slices, current_exposures, n_exposures)

        # Calculate normalized loss
        raw_importance_current = v_optimal - q_min_current
        goodness = tf.math.divide_no_nan(q_predicted - q_min_current, raw_importance_current)
        goodness_clipped = tf.clip_by_value(goodness, 0.0, 1.0) # Clip for numerical stability

        reward = goodness_clipped * importance_percentile
        loss = 1.0 - reward

        return tf.reduce_mean(loss)

    return suboptimality_loss

class ForexData:

    __slots__ = ("train_data", "eval_data", "train_features", "eval_features", "feature_names")

    def __init__(self, fcd: ForexCandleData, feature_names: list[str], split: float):

        logging.info("Computing features...")
        fe = FeatureEngineer(fcd.df)
        data = fe.get_all(MarketDataCol.all_names())
        features = fe.get_all(feature_names)

        logging.info("Cropping initial NaNs...")
        crop_idx = max(find_first_valid_row(data), find_first_valid_row(features))
        data = data[crop_idx:]
        features = features[crop_idx:]
        assert len(data) == len(features)
        assert data.ndim == 2 and features.ndim == 2
        assert not contains_nan_or_inf(data)
        assert not contains_nan_or_inf(features)

        logging.info("Splitting the data and features...")
        split_idx = int(len(data) * split)
        self.train_data = data[:split_idx]
        self.eval_data = data[split_idx:]
        self.train_features = features[:split_idx]
        self.eval_features = features[split_idx:]
        self.feature_names = feature_names



def train_model(forex_data: ForexData):

    data = forex_data.train_data
    N = data.shape[0] - 2
    features = forex_data.train_features[:N]

    logging.info("Setting up directories...")
    model_name = "supervised_fx_suboptimality"
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Splitting training data for validation...")
    train_features, val_features, train_indices, val_indices = train_test_split(
        features, np.arange(N), test_size=0.2, random_state=42, shuffle=True
    )

    logging.info("Setting up model...")
    train_generator = SuboptimalityDataGenerator(train_features, 1024, train_indices)
    validation_generator = SuboptimalityDataGenerator(val_features, 1024, val_indices)

    input_dim = train_generator.X.shape[1]
    features_input = Input(shape=(input_dim,), name="features_input")
    x = Dense(48, activation='relu')(features_input)
    x = Dense(48, activation='relu')(x)
    predicted_exposure = Dense(1, activation='tanh', name="output_exposure")(x)

    model = Model(inputs=features_input, outputs=predicted_exposure)
    model.summary()

    dp_table = get_dp_table(data, TRANSACTION_COST_PCT, 15, 15, DP_CACHE_DIR)
    loss_fn = create_suboptimality_loss(dp_table, data, TRANSACTION_COST_PCT)

    model.compile(optimizer=Adam(learning_rate=3e-4), loss=loss_fn)

    model_path = model_dir / f"model.keras"
    model_checkpoint = ModelCheckpoint(str(model_path), monitor='val_loss', save_best_only=True, mode='min')

    log_dir = model_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    logging.info("Training model...")
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=5_000,
        callbacks=[model_checkpoint, tensorboard_callback, TqdmCallback(verbose=0)],
        verbose=0
    )
    logging.info(f"Training finished. Best model saved to {model_path}")

class SupervisedModelWrapper(DummyModel):
    def __init__(self, model_path: Path):
        super().__init__()
        assert model_path.exists()
        assert model_path.suffix == ".keras"
        custom_objects = {"suboptimality_loss": dummy_loss}
        self.model = load_model(model_path, custom_objects=custom_objects, compile=False)

    def predict(self, obs: Union[np.ndarray, dict[str, np.ndarray]], *args, **kwargs) -> tuple[np.ndarray, None]:
        obs_tensor = tf.convert_to_tensor(np.atleast_2d(obs), dtype=tf.float32)
        prediction_tensor = self.model(obs_tensor, training=False)
        return prediction_tensor.numpy(), None

def evaluate_model(forex_data: ForexData, model_path: Optional[Path] = None):

    if model_path is None:
        model_path = max((p for p in MODELS_DIR.rglob("*.keras")), key=lambda x: x.stat().st_mtime)
    assert model_path.exists()
    model_dir = model_path.resolve().parent

    logging.info(f"Evaluating: {model_path}")

    logging.info("Setting up environments...")
    sfe = StepwiseFeatureEngineer()
    sfe.add(["current_exposure"], calculate_current_exposure)

    action_config = ActionConfig(n=0, low=-1.0, high=1.0)
    env_config = EnvConfig(transaction_cost_pct=TRANSACTION_COST_PCT)
    train_obs = EnvObs(features_data=forex_data.train_features,
                       feature_names=forex_data.feature_names,
                       sfe=sfe,
                       name="obs_vector",
                       window=1)
    eval_obs = EnvObs(features_data=forex_data.eval_features,
                      feature_names=forex_data.feature_names,
                      sfe=sfe,
                      name="obs_vector",
                      window=1)
    train_data_config = DataConfig(forex_data.train_data, [train_obs])
    eval_data_config = DataConfig(forex_data.eval_data, [eval_obs])
    train_env = ForexEnv(action_config, env_config, train_data_config)
    eval_env = ForexEnv(action_config, env_config, eval_data_config)

    logging.info("Loading model")
    model = SupervisedModelWrapper(model_path)

    logging.info("Running evaluation on the full training dataset...")
    train_run_path = model_dir / "train" / "data.csv"
    run_model(model, train_env, train_run_path)
    analyse_individual_run(train_run_path, "train")
    logging.info(f"Training set evaluation complete. Results in {train_run_path.parent}")

    logging.info("Running evaluation on the full evaluation dataset...")
    eval_run_path = model_dir / "eval" / "data.csv"
    run_model(model, eval_env, eval_run_path)
    analyse_individual_run(eval_run_path, "eval")
    logging.info(f"Evaluation set evaluation complete. Results in {eval_run_path.parent}")


if __name__ == "__main__":

    fcd = ForexCandleData.load("dukascopy", "EURUSD", Timeframe.H1, datetime(2020, 1, 1, 22), datetime(2024, 12, 31, 21))
    feature_names = [
        "sin_24h",
        "cos_24h",
        "cos_7d",
        "sin_7d",
        "as_ratio_of_other_column('parabolic_sar(0.02, 0.2)', 'close_bid')",
        "as_ratio_of_other_column('ema(24)', 'close_bid')",
        "as_ratio_of_other_column('ema(72)', 'close_bid')",
        "as_min_max_fixed('adx(14)', 0, 100)",
        "as_min_max_fixed('rsi(14)', 0, 100)",
        "as_z_score('macd_hist(12, 26, 9)', 50)",
        "as_min_max_fixed('stoch_k(14)', 0, 100)",
        "as_ratio_of_other_column('atr(14)', 'close_bid')",
        "as_ratio_of_other_column('bb_upper(20, 2.0)', 'close_bid')",
        "as_ratio_of_other_column('bb_lower(20, 2.0)', 'close_bid')",
        "as_z_score(\"as_ratio_of_other_column('close_ask', 'close_bid')\", 50)"
    ]
    fd = ForexData(fcd, feature_names, 0.7)
    train_model(fd)
    evaluate_model(fd)
