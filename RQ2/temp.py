import logging

import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import LeakyReLU

from RQ2.constants import *
from RQ2.parameters import INITIAL_CAPITAL, TRANSACTION_COST_PCT, SEED
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import StepwiseFeatureEngineer, get_current_exposure, duration_of_current_trade 
from common.envs.forex_env import ForexEnv
from common.models.train_eval import run_experiment
from common.envs.rewards import percentage_return
from common.scripts import *


def get_feature_engineers() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    """
    Create and return the feature engineer and stepwise feature engineer objects.
    """
    # Create a feature engineer object
    feature_engineer = FeatureEngineer()

    looky_backy = 4

    # ------------------------- #
    # ---- TIME Indicators ---- #
    # ------------------------- #
    feature_engineer.add(complex_24h) # 2


    # ----------------------------- #
    # ---- TA-Trend Indicators ---- #
    # ----------------------------- #
    def feat_sar(df):
        parabolic_sar(df)
        as_ratio_of_other_column(df, 'sar', 'close_bid')
        history_lookback(df, looky_backy, ["sar"])
    feature_engineer.add(feat_sar) # 1 * looky_backy
    
    def feat_vwap(df):
        vwap(df)
        as_ratio_of_other_column(df, 'vwap_14', 'close_bid')
        history_lookback(df, looky_backy, ["sar"])
    feature_engineer.add(feat_vwap) # 1 * looky_backy

    # -------------------------------- #
    # ---- TA-Momentum Indicators ---- #
    # -------------------------------- #

    def feat_macd(df):
        macd(df, short_window=12, long_window=26, signal_window=9)
        remove_columns(df, ["macd_signal", "macd"])
        as_z_score(df, 'macd_hist', window=50)
        history_lookback(df, looky_backy, ["macd_hist"])
    feature_engineer.add(feat_macd) # 1

    def feat_mfi(df):
        mfi(df)
        as_min_max_fixed(df, 'mfi_14', 0, 100)
    feature_engineer.add(feat_mfi) # 1 * looky_backy

    # ---------------------------------- #
    # ---- TA-Volatility Indicators ---- #
    # ---------------------------------- #

    def feat_boll_bands(df):
        bollinger_bands(df, window=20, num_std_dev=2)
        as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
        as_ratio_of_other_column(df, "bb_lower_20", "close_bid")
        history_lookback(df, looky_backy, ["bb_upper_20"])
        history_lookback(df, looky_backy, ["bb_lower_20"])
    feature_engineer.add(feat_boll_bands) # 2 * looky_backy

    def feat_ch_vol(df):
        chaikin_volatility(df)
        history_lookback(df, looky_backy, ['chaikin_vol_10_10'])
    feature_engineer.add(feat_ch_vol) # 1 * looky_backy

    # -------------------------- #
    # ---- Agent Indicators ---- #
    # -------------------------- #

    # Create a stepwise feature engineer object
    stepwise_feature_engineer = StepwiseFeatureEngineer()
    stepwise_feature_engineer.add(['cash_percentage'], get_current_exposure) # 1
    stepwise_feature_engineer.add(['current_trade_length'], duration_of_current_trade) # 1

    return feature_engineer, stepwise_feature_engineer



def main():
    
    set_seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    forex_data = ForexCandleData.load(source="dukascopy",
                                      instrument="EURUSD",
                                      granularity=Timeframe.M15,
                                      start_time=RQ2_HYPERPARAMETERS_START_DATE,
                                      end_time= RQ2_HYPERPARAMETERS_END_DATE_15M,
                                    )
    
    # --- Feature Engineering ---
    
    feature_engineer, stepwise_feature_engineer = get_feature_engineers()

    logging.info("Creating environments...")
    train_env, eval_env = ForexEnv.create_split_envs(
        split_pcts=[RQ2_DATA_SPLIT_RATIO, 1 - RQ2_DATA_SPLIT_RATIO],
        forex_candle_data=forex_data,
        market_feature_engineer=feature_engineer,
        agent_feature_engineer=stepwise_feature_engineer,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        n_actions=3,
        custom_reward_function=percentage_return
        )
    logging.info("Environments created.")

    temp_env = DummyVecEnv([lambda: train_env])

    experiment_funcs: List[Callable[[DummyVecEnv], DQN]] = [
        exprmt_aggresive,
        exprmt_balanced,
        exprmt_cautious,
        exprmt_patient
    ]

    for experiment_func in experiment_funcs:
        logging.info(f"Running experiment: {experiment_func.__name__}")
        dqn_model = experiment_func(temp_env)
        logging.info("Running train test analyze...")
        run_experiment(
            train_env=train_env,
            validate_env=eval_env,
            model=dqn_model,
            base_folder_path=RQ2_DIR,
            experiment_group_name="[hyperparameters]-15m_data",
            experiment_name=experiment_func.__name__,
            train_episodes=250,
            eval_episodes=1,
            checkpoints=True,
            tensorboard_logging=True,
        )
        logging.info(f"Experiment {experiment_func.__name__} completed.")
    

# EXPERIMENT FUNCTIONS

def base_experiment_func(temp_env: DummyVecEnv) -> Dict[str, Any]:
    policy_kwargs = dict(net_arch=[32, 16], optimizer_class=optim.Adam, activation_fn=LeakyReLU)

    kwargs = dict(
        policy="MlpPolicy",
        env=temp_env,
        learning_starts=1000,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=SEED
    )
    return kwargs

def exprmt_patient(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'patient' DQN model.
    """
    dqn_kwargs = base_experiment_func(temp_env)

    dqn_kwargs['learning_rate'] = 0.00001
    dqn_kwargs['buffer_size'] = 100_000
    dqn_kwargs['batch_size'] = 512
    dqn_kwargs['tau'] = 0.001
    dqn_kwargs['train_freq'] = (1, 'episode')
    dqn_kwargs['gradient_steps'] = -1
    dqn_kwargs['target_update_interval'] = 5_000
    dqn_kwargs['exploration_fraction'] = 0.5
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.01

    return DQN(**dqn_kwargs)

def exprmt_cautious(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'cautious' DQN model.
    """
    dqn_kwargs = base_experiment_func(temp_env)

    dqn_kwargs['learning_rate'] = 0.00005
    dqn_kwargs['buffer_size'] = 60_000
    dqn_kwargs['batch_size'] = 512
    dqn_kwargs['tau'] = 0.0025
    dqn_kwargs['train_freq'] = 64
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 2500
    dqn_kwargs['exploration_fraction'] = 0.4
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.02

    return DQN(**dqn_kwargs)

def exprmt_balanced(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'balanced' DQN model.
    """
    dqn_kwargs = base_experiment_func(temp_env)

    dqn_kwargs['learning_rate'] = 0.0001
    dqn_kwargs['buffer_size'] = 30_000
    dqn_kwargs['batch_size'] = 256
    dqn_kwargs['tau'] = 0.005
    dqn_kwargs['train_freq'] = 16
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 1_000
    dqn_kwargs['exploration_fraction'] = 0.33
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.05

    return DQN(**dqn_kwargs)

def exprmt_aggresive(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes an 'aggressive' DQN model.
    """
    dqn_kwargs = base_experiment_func(temp_env)

    dqn_kwargs['learning_rate'] = 0.001
    dqn_kwargs['buffer_size'] = 5_000
    dqn_kwargs['batch_size'] = 64
    dqn_kwargs['tau'] = 0.01
    dqn_kwargs['train_freq'] = 4
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 500
    dqn_kwargs['exploration_fraction'] = 0.25
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.1

    return DQN(**dqn_kwargs)



if __name__ == '__main__':
    main()