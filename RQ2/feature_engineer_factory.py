import logging

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from common.data.feature_engineer import *
from common.data.stepwise_feature_engineer import *

class RQ2FeatureEngineerFactory:
    """
    Factory class to create feature engineers for RQ2 experiments.
    """

    def __init__(self, looky_backy: int = 4):
        self._feature_engineer = FeatureEngineer()
        self._stepwise_feature_engineer = StepwiseFeatureEngineer()
        self._looky_backy = looky_backy

    @classmethod
    def create_core_factory(cls, looky_backy: int = 4,
                                 time: bool = True,
                                 trend: bool = True,
                                 momentum: bool = True,
                                 volatility: bool = True,
                                 agent: bool = True) -> 'RQ2FeatureEngineerFactory':
        factory = cls(looky_backy)
        if time:
            factory.add_sin_24h()
        if trend:
            factory.add_parabolic_sar()
        if momentum:
            factory.add_macd()
        if volatility:
            factory.add_bollinger_bands()
        if agent:
            factory.add_current_exposure()
        return factory
    
    def give_me_them_engineers(self) -> tuple[FeatureEngineer, StepwiseFeatureEngineer]:
        return self._feature_engineer, self._stepwise_feature_engineer
    
    # TIME FEATURES
    def add_lin_24h(self):
        self._feature_engineer.add(lin_24h)
        return self
    
    def add_lin_7d(self):
        self._feature_engineer.add(lin_7d)
        return self

    def add_sin_24h(self):
        self._feature_engineer.add(sin_24h)
        return self
    
    def add_cos_24h(self):
        self._feature_engineer.add(cos_24h)
        return self
    
    def add_sin_7d(self):
        self._feature_engineer.add(sin_7d)
        return self
    
    def add_cos_7d(self):
        self._feature_engineer.add(cos_7d)
        return self

    # TREND FEATURES
    def add_parabolic_sar(self):
        def feat_sar(df):
            parabolic_sar(df)
            as_ratio_of_other_column(df, 'sar', 'close_bid')
            history_lookback(df, self._looky_backy, ["sar"])
        self._feature_engineer.add(feat_sar)
        return self
    
    def add_vwap(self):
        def feat_vwap(df):
            vwap(df)
            as_ratio_of_other_column(df, 'vwap_14', 'close_bid')
            history_lookback(df, self._looky_backy, ["vwap_14"])
        self._feature_engineer.add(feat_vwap)
        return self
    
    def add_kama(self):
        def feat_kama(df):
            kama(df)
            as_ratio_of_other_column(df, 'kama_10_close_bid', 'close_bid')
            history_lookback(df, self._looky_backy, ["kama_10_close_bid"])
        self._feature_engineer.add(feat_kama)
        return self

    # MOMENTUM FEATURES
    def add_macd(self):
        def feat_macd(df):
            macd(df, short_window=12, long_window=26, signal_window=9)
            remove_columns(df, ["macd_signal", "macd"])
            as_z_score(df, 'macd_hist', window=50)
            history_lookback(df, self._looky_backy, ["macd_hist"])
        self._feature_engineer.add(feat_macd)
        return self

    def add_mfi(self):
        def feat_mfi(df):
            mfi(df)
            as_min_max_fixed(df, 'mfi_14', 0, 100)
            history_lookback(df, self._looky_backy, ["mfi_14"])
        self._feature_engineer.add(feat_mfi)
        return self
    
    def add_cci(self):
        def feat_cci(df):
            cci(df, window=20)
            as_min_max_fixed(df, 'cci_20', -100, 100)
            history_lookback(df, self._looky_backy, ["cci_20"])
        self._feature_engineer.add(feat_cci)
        return self

    # VOLATILITY FEATURES
    def add_bollinger_bands(self):
        def feat_boll_bands(df):
            bollinger_bands(df, window=20, num_std_dev=2)
            as_ratio_of_other_column(df, "bb_upper_20", "close_bid")
            as_ratio_of_other_column(df, "bb_lower_20", "close_bid")
            history_lookback(df, self._looky_backy, ["bb_upper_20"])
            history_lookback(df, self._looky_backy, ["bb_lower_20"])
        self._feature_engineer.add(feat_boll_bands)
        return self

    def add_eom(self):
        def feat_eom(df):
            ease_of_movement(df, window=14)
            as_min_max_window(df, 'eom_14', window=50)
            history_lookback(df, self._looky_backy, ["eom_14"])
        self._feature_engineer.add(feat_eom)
        return self
    
    def add_atr(self):
        def feat_atr(df):
            atr(df, window=14)
            as_z_score(df, 'atr_14', window=50)
            history_lookback(df, self._looky_backy, ["atr_14"])
        self._feature_engineer.add(feat_atr)
        return self

    # AGENT FEATURES
    def add_current_exposure(self):
        self._stepwise_feature_engineer.add(["current_exposure"], get_current_exposure)
        return self
    
    def add_duration_of_current_trade(self):
        self._stepwise_feature_engineer.add(["duration_of_trade"], duration_of_current_trade)
        return self
