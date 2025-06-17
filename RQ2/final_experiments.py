import logging

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import *

from RQ2.feature_engineer_factory import RQ2FeatureEngineerFactory

# ########################
# # -- Sub-Question 1 -- #
# ########################

# ----------------- #
# - Time Features - #
# ----------------- #

def S1_TM_NONE() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    return factory.give_me_them_engineers()

def S1_TM_L24() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    factory.add_lin_24h()

def S1_TM_S24() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    factory.add_sin_24h()
    return factory.give_me_them_engineers()

def S1_TM_SC24() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    factory.add_sin_24h().add_cos_24h()
    return factory.give_me_them_engineers()

def S1_TM_L24L7() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    factory.add_lin_24h().add_lin_7d()
    return factory.give_me_them_engineers()

def S1_TM_SC24SC7() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    factory.add_sin_24h().add_cos_24h().add_sin_7d().add_cos_7d()
    return factory.give_me_them_engineers()

def S1_TM_COMBO() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    factory.add_lin_24h().add_sin_24h().add_cos_24h()
    return factory.give_me_them_engineers()

def S1_TM_ALL() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(time=False)
    factory.add_lin_24h().add_lin_7d().add_sin_24h().add_cos_24h().add_sin_7d().add_cos_7d()
    return factory.give_me_them_engineers()

# ------------------ #
# - Trend Features - #
# ------------------ #

def S1_TR_NONE() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(trend=False)
    return factory.give_me_them_engineers()

def S1_TR_NV() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(trend=False)
    factory.add_parabolic_sar()
    return factory.give_me_them_engineers()

def S1_TR_V() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(trend=False)
    factory.add_vwap()
    return factory.give_me_them_engineers()

def S1_TR_COMBO() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(trend=False)
    factory.add_parabolic_sar().add_vwap()
    return factory.give_me_them_engineers()

def S1_TR_ALL() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(trend=False)
    factory.add_parabolic_sar().add_vwap().add_kama()
    return factory.give_me_them_engineers()

# --------------------- #
# - Momentum Features - #
# --------------------- #

def S1_MO_NONE() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(momentum=False)
    return factory.give_me_them_engineers()

def S1_MO_NV() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(momentum=False)
    factory.add_macd()
    return factory.give_me_them_engineers()

def S1_MO_V() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(momentum=False)
    factory.add_mfi()
    return factory.give_me_them_engineers()

def S1_MO_COMBO() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(momentum=False)
    factory.add_macd().add_mfi()
    return factory.give_me_them_engineers()

def S1_MO_ALL() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(momentum=False)
    factory.add_macd().add_mfi().add_cci()
    return factory.give_me_them_engineers()

# ----------------------- #
# - Volatility Features - #
# ----------------------- #

def S1_VO_NONE() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(volatility=False)
    return factory.give_me_them_engineers()

def S1_VO_NV() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(volatility=False)
    factory.add_bollinger_bands()
    return factory.give_me_them_engineers()

def S1_VO_V() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(volatility=False)
    factory.add_eom()
    return factory.give_me_them_engineers()

def S1_VO_COMBO() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(volatility=False)
    factory.add_bollinger_bands().add_eom()
    return factory.give_me_them_engineers()

def S1_VO_ALL() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(volatility=False)
    factory.add_bollinger_bands().add_eom().add_atr()
    return factory.give_me_them_engineers()

# ------------------ #
# - Agent Features - #
# ------------------ #

def S1_AG_NONE() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(agent=False)
    return factory.give_me_them_engineers()

def S1_AG_CE() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(agent=False)
    factory.add_current_exposure()
    return factory.give_me_them_engineers()

def S1_AG_DT() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]: 
    factory = RQ2FeatureEngineerFactory.create_core_factory(agent=False)
    factory.add_duration_of_current_trade()
    return factory.give_me_them_engineers()

def S1_AG_ALL() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory(agent=False)
    factory.add_current_exposure().add_duration_of_current_trade()
    return factory.give_me_them_engineers()

# --------------------------- #
# - Combinatory Experiments - #
# --------------------------- #

def S1_COMBO_COMBO() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory()
    factory.add_lin_24h().add_sin_24h().add_cos_24h() \
           .add_parabolic_sar().add_vwap() \
           .add_macd().add_mfi() \
           .add_bollinger_bands().add_eom() \
           .add_current_exposure().add_duration_of_current_trade()
    return factory.give_me_them_engineers()

def S1_COMBO_ALL() -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
    factory = RQ2FeatureEngineerFactory.create_core_factory()
    factory.add_lin_24h().add_lin_7d().add_sin_24h().add_cos_24h().add_sin_7d().add_cos_7d() \
           .add_parabolic_sar().add_vwap().add_kama() \
           .add_macd().add_mfi().add_cci() \
           .add_bollinger_bands().add_eom().add_atr() \
           .add_current_exposure().add_duration_of_current_trade()
    return factory.give_me_them_engineers()

# ########################
# # -- Sub-Question 2 -- #
# ########################

