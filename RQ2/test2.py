import logging
import random

from RQ2.constants import LOGS_DIR, MODELS_DIR
from common.scripts import run_model_on_vec_env
import numpy as np
from common.scripts import combine_df
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from common.env.forex_env import DiscreteActionForexEnv
from common.feature.feature_engineer import FeatureEngineer
from common.feature.stepwise_feature_engineer import StepwiseFeatureEngineer

from common.data import ForexData
from common.constants import *
from common.scripts import *

if __name__ == '__main__':
   

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration Parameters ---
    INITIAL_CAPITAL = 10000.0
    TRANSACTION_COST_PCT = 0.0 # Example: 0.1% commission per trade

    # set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Get ask and bid data, and combine
    ask_path = FOREX_DIR / "EURUSD" / "15M" / "ASK" / "10.05.2022T00.00-10.05.2025T23.45.csv"
    bid_path = FOREX_DIR / "EURUSD" / "15M" / "BID" / "10.05.2022T00.00-10.05.2025T23.45.csv"
    ask_df = ForexData(ask_path).df
    bid_df = ForexData(ask_path).df
    forex_data = combine_df(bid_df, ask_df)
    # forex_data = filter_df(forex_data)
    train_df, eval_df = split_df(forex_data, 0.7)

    # --- Feature Engineering ---
    # Create a feature engineer object
    feature_engineer = FeatureEngineer()
    # Add feature engineering steps
    def create_price_columns(df):
        """
        Create price columns for the DataFrame.
        """
        df['price'] = df['close_bid']
        return df
    
    def rsi(df, window=14):
        """
        Calculate the Relative Strength Index (RSI) column.
        """
        delta = df['close_bid'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        
        # replace rs with 1 if NaN, 100 if rs is np.inf, or 0 if rs is -np.inf
        rs = rs.fillna(1)
        rs = rs.replace(np.inf, 100)
        rs = rs.replace(-np.inf, 0)


        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi
        return df

    feature_engineer.add(rsi)

    # Add stepwise feature engineering
    stepwise_feature_engineer = StepwiseFeatureEngineer(columns=['cash_percentage'])
    def calculate_cash_percentage(df, index):
        """
        Calculate the cash to shares ratio.
        """
        current_cash = df['cash'].iloc[index]
        current_equity = df['equity_close'].iloc[index]
        percentage = current_cash / current_equity
        return {'cash_percentage': percentage}
    stepwise_feature_engineer.add(calculate_cash_percentage)

    logging.info("Creating training environment...")
    train_env = DummyVecEnv([lambda: DiscreteActionForexEnv(
        market_data_df=train_df,
        data_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
    )])
    logging.info("Training environment created.")

    policy_kwargs = dict(net_arch=[3])

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.001,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42,
    )

    logging.info("Training the DQN agent...")
    model.learn(total_timesteps=1_000, progress_bar=True)
    logging.info("Training finished.")

    logging.info("Saving the DQN model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{timestamp}_DQN_model"
    model_path = MODELS_DIR / model_name
    model.save(model_path)
    logging.info(f"Model saved to {model_path}.")

    logging.info("\nEvaluating the agent on the eval_df...")
    eval_env = DummyVecEnv([
        lambda: DiscreteActionForexEnv(
            market_data_df=eval_df,
            data_feature_engineer=feature_engineer,
            agent_feature_engineer=stepwise_feature_engineer,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost_pct=TRANSACTION_COST_PCT,
        )
    ])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{timestamp}_DQN_model"
    run_model_on_vec_env(model, eval_env, log_path)

