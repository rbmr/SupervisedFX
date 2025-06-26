import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, InputLayer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Assuming your project structure allows these imports
# If running as a standalone script, you might need to adjust sys.path
from src.constants import MODEL_DIR, DP_CACHE_DIR, AgentDataCol
from src.data.data import ForexCandleData, Timeframe
from src.data.feature_engineer import FeatureEngineer
from src.data.stepwise_feature_engineer import StepwiseFeatureEngineer
from src.envs.dp import DPTable, get_dp_table, get_optimal_action
from src.envs.forex_env import ForexEnv
from src.models.analysis import analyse_individual_run
from src.models.dummy_models import DummyModel
from src.models.train_eval import run_model

# --- Configuration ---
TRANSACTION_COST_PCT = 5 / 100_000
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Custom Keras Data Generator ---
class OptimalPolicyGenerator(Sequence):
    """
    Generates data for Keras on-the-fly. For each epoch, it re-samples
    random exposures and creates a new shuffled (X, y) dataset.
    """

    def __init__(self, market_features: np.ndarray, dp_table: DPTable, batch_size: int, global_indices: np.ndarray):
        super().__init__()
        self.market_features = market_features
        self.dp_table = dp_table
        self.batch_size = batch_size
        self.global_indices = global_indices  # Original indices in the full DP table
        self.n_samples = self.market_features.shape[0]
        self.X = None
        self.y = None
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        return self.X[start_idx:end_idx], self.y[start_idx:end_idx]

    def on_epoch_end(self):
        """Generate a new (X, y) dataset at the end of each epoch."""
        logging.info(f"{self.__class__.__name__} ({self.n_samples} samples): Regenerating data for new epoch...")
        random_exposures = np.random.uniform(-1.0, 1.0, self.n_samples)
        y_optimal_actions = np.array([
            get_optimal_action(self.dp_table, self.global_indices[t], random_exposures[t])
            for t in range(self.n_samples)
        ]).reshape(-1, 1)
        X_full_features = np.column_stack((self.market_features, random_exposures))
        self.X, self.y = shuffle(X_full_features, y_optimal_actions)


# --- Model Wrapper for Evaluation ---
class SupervisedModel(DummyModel):
    """
    Wraps the trained Keras model to make it compatible with the evaluation pipeline.
    """

    def __init__(self, env: ForexEnv, model_path: Path):
        self.model = load_model(str(model_path))
        self.env = env
        super().__init__(pred_fn=self._predict_action)

    def _predict_action(self, obs: np.ndarray) -> float:
        """Constructs the full feature vector and predicts the target exposure."""
        # The observation from the env should only contain market features.
        # We calculate the current agent exposure here.
        if self.env.n_steps == 0:
            current_exposure = 0.0
        else:
            pre_action_equity = self.env.agent_data[self.env.n_steps, AgentDataCol.pre_action_equity]
            cash = self.env.agent_data[self.env.n_steps - 1, AgentDataCol.cash]
            current_exposure = (pre_action_equity - cash) / pre_action_equity if pre_action_equity != 0 else 0.0

        model_input = np.append(obs, current_exposure).reshape(1, -1)
        prediction = self.model.predict(model_input, verbose=0)
        return float(prediction.item())


# --- Main Pipeline Functions ---
def prepare_data() -> tuple[np.ndarray, np.ndarray, ForexEnv, ForexEnv, DPTable]:
    """Handles all data loading, feature engineering, and splitting."""
    logging.info("Step 1 & 2: Fetching data and computing features...")
    fcd = ForexCandleData.load("dukascopy", "EURUSD", Timeframe.H1, datetime(2020, 1, 1), datetime(2024, 12, 31))

    fe = FeatureEngineer(fcd.df)
    features = (
        "sin_24h", "cos_24h", "cos_7d", "sin_7d",
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
    )

    logging.info("Step 3: Cropping initial NaNs...")
    # For evaluation, we need environments that only use market features, not stepwise features.
    sfe_empty = StepwiseFeatureEngineer(feature_configs=[])  # No stepwise features

    full_train_env, full_eval_env = ForexEnv.create_split_envs(
        split_pcts=[0.7, 0.3],
        forex_candle_data=fcd,
        market_feature_engineer=fe,
        agent_feature_engineer=sfe_empty,
        n_actions=0,
        transaction_cost_pct=TRANSACTION_COST_PCT
    )

    logging.info("Step 4: Computing DP table on full dataset...")
    full_market_data_df = pd.concat([full_train_env.market_data, full_eval_env.market_data], ignore_index=True)
    dp_table = get_dp_table(full_market_data_df.to_numpy(), TRANSACTION_COST_PCT, 15, 30, DP_CACHE_DIR)

    logging.info("Step 5 & 6: Splitting training data for validation...")
    full_train_features = full_train_env.observations[0].features_data
    full_train_indices = np.arange(len(full_train_features))

    train_features, val_features, train_indices, val_indices = train_test_split(
        full_train_features, full_train_indices, test_size=0.2, random_state=42, shuffle=True
    )

    return train_features, val_features, train_indices, val_indices, full_train_env, full_eval_env, dp_table


def create_and_train(train_features, val_features, train_indices, val_indices, dp_table) -> Path:
    """Step 7: Creates and trains the model using data generators."""
    logging.info("\n--- Step 7: Starting Supervised Model Training ---")

    train_generator = OptimalPolicyGenerator(train_features, dp_table, 256, train_indices)
    validation_generator = OptimalPolicyGenerator(val_features, dp_table, 256, val_indices)

    input_dim = train_generator.X.shape[1]
    model = Sequential([
        InputLayer(input_shape=(input_dim,), name="input_layer"),
        Dense(32, activation='sigmoid', name="hidden_layer_1"),
        Dense(32, activation='sigmoid', name="hidden_layer_2"),
        Dense(1, activation='tanh', name="output_exposure")
    ])
    model.summary()
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min')
    model_name = f"supervised_fx_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_path = MODEL_DIR / f"{model_name}.keras"
    model_checkpoint = ModelCheckpoint(str(model_path), monitor='val_loss', save_best_only=True, mode='min')

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=500,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    logging.info(f"Training finished. Best model saved to {model_path}")
    return model_path, model_name


def evaluate_final_model(model_path: Path, model_name: str, full_train_env: ForexEnv, full_eval_env: ForexEnv):
    """Step 8 & 9: Evaluates the best model on the full train and eval sets."""
    logging.info(f"\n--- Step 8 & 9: Final Evaluation for Model: {model_name} ---")

    # --- Run on Full Training Set ---
    logging.info("Running evaluation on the full (70%) training dataset (chronological)...")
    train_run_path = MODEL_DIR / model_name / "final_train_run.csv"
    train_run_path.parent.mkdir(exist_ok=True)

    train_model_wrapper = SupervisedModel(env=full_train_env, model_path=model_path)
    run_model(train_model_wrapper, full_train_env, train_run_path, full_train_env.episode_len, True)
    analyse_individual_run(train_run_path, f"{model_name}-TrainSet")
    logging.info(f"Training set evaluation complete. Results in {train_run_path.parent}")

    # --- Run on Full Evaluation Set ---
    logging.info("Running evaluation on the full (30%) evaluation dataset (chronological)...")
    eval_run_path = MODEL_DIR / model_name / "final_eval_run.csv"

    eval_model_wrapper = SupervisedModel(env=full_eval_env, model_path=model_path)
    run_model(eval_model_wrapper, full_eval_env, eval_run_path, full_eval_env.episode_len, True)
    analyse_individual_run(eval_run_path, f"{model_name}-EvalSet")
    logging.info(f"Evaluation set evaluation complete. Results in {eval_run_path.parent}")

if __name__ == "__main__":
    # Execute the full pipeline
    train_feats, val_feats, train_idx, val_idx, train_env, eval_env, dp_tbl = prepare_data()
    best_model_path, trained_model_name = create_and_train(train_feats, val_feats, train_idx, val_idx, dp_tbl)
    evaluate_final_model(best_model_path, trained_model_name, train_env, eval_env)