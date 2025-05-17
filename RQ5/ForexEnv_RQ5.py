import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stockstats import StockDataFrame

INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 0.001  # 0.1%
LOOKBACK_WINDOW_SIZE = 30

RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

class ForexEnv(gym.Env):
    """
    Custom Forex trading environment for reinforcement learning.
    Actions: 0 - HOLD, 1 - LONG, 2 - SHORT
    Observation: concatenated past N market states + current portfolio state
    """

    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, df, initial_capital=INITIAL_CAPITAL, transaction_cost_pct=TRANSACTION_COST_PCT,
                 lookback_window_size=LOOKBACK_WINDOW_SIZE, log_level=0, seed=None):
        super(ForexEnv, self).__init__()

        self.df = df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_window_size = lookback_window_size
        self.log_level = log_level
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.num_market_features = 6
        obs_size = self.lookback_window_size * self.num_market_features + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._prepare_data()
        self.reset()

    def _prepare_data(self):
        df = self.df.copy()

        df['mid_close'] = (df['close_bid'] + df['close_ask']) / 2
        df['mid_open'] = (df['open_bid'] + df['open_ask']) / 2
        df['mid_high'] = (df['high_bid'] + df['high_ask']) / 2
        df['mid_low'] = (df['low_bid'] + df['low_ask']) / 2

        df['log_ret_bid'] = np.log(df['close_bid'] / df['close_bid'].shift(1)).fillna(0)
        df['log_ret_ask'] = np.log(df['close_ask'] / df['close_ask'].shift(1)).fillna(0)
        df['spread'] = df['close_ask'] - df['close_bid']
        df['norm_spread'] = df['spread'] / df['mid_close']

        if 'volume' in df.columns:
            vol_min = df['volume'].rolling(window=self.lookback_window_size, min_periods=1).min()
            vol_max = df['volume'].rolling(window=self.lookback_window_size, min_periods=1).max()
            denom = vol_max - vol_min
            df['norm_volume'] = pd.Series(np.where(denom == 0, 0.5, (df['volume'] - vol_min) / denom), index=df.index).fillna(0.5)
        else:
            df['norm_volume'] = 0.5

        stock_df = pd.DataFrame({
            'open': df['mid_open'], 'high': df['mid_high'], 'low': df['mid_low'],
            'close': df['mid_close'], 'volume': df.get('volume', 0.0)
        })

        stock_sdf = StockDataFrame.retype(stock_df)
        df['rsi'] = stock_sdf[f'rsi_{RSI_PERIOD}']
        df['macd_diff'] = stock_sdf['macdh']

        for col in ['rsi', 'macd_diff']:
            min_val, max_val = df[col].min(), df[col].max()
            df[f'norm_{col}'] = ((df[col] - min_val) / (max_val - min_val)).fillna(0.5) if max_val > min_val else 0.5

        self.processed_df = df[[
            'log_ret_bid', 'log_ret_ask', 'norm_spread', 'norm_volume', 'norm_rsi', 'norm_macd_diff',
            'close_bid', 'close_ask']].fillna(0)
        self.total_steps = len(self.processed_df) - self.lookback_window_size - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0.0
        self.position = 0  # 0 = CASH, 1 = LONG, -1 = SHORT
        self.equity = self.initial_capital
        self.history = {'equity': [self.equity], 'actions': []}
        return self._get_observation(), {}

    def _get_observation(self):
        i = self.current_step
        market_window = self.processed_df.iloc[i:i+self.lookback_window_size][[
            'log_ret_bid', 'log_ret_ask', 'norm_spread', 'norm_volume', 'norm_rsi', 'norm_macd_diff']].values.flatten()
        bid, ask = self._get_current_prices()
        equity = self._calculate_equity(bid, ask)
        cash_ratio = np.clip(self.cash / equity, -1e6, 1e6) if equity > 0 else 0.0
        long_flag = 1.0 if self.position == 1 else 0.0
        short_flag = 1.0 if self.position == -1 else 0.0
        return np.concatenate((market_window, [cash_ratio, long_flag, short_flag])).astype(np.float32)

    def _get_current_prices(self):
        idx = self.current_step + self.lookback_window_size
        idx = min(idx, len(self.processed_df) - 1)
        return self.processed_df.iloc[idx]['close_bid'], self.processed_df.iloc[idx]['close_ask']

    def _calculate_equity(self, bid, ask):
        if self.position == 1:
            return self.cash + self.shares * bid
        elif self.position == -1:
            return self.cash - abs(self.shares) * ask
        return self.cash

    def step(self, action):
        self.current_step += 1
        bid, ask = self._get_current_prices()
        prev_equity = self.equity
        self._trade(action, bid, ask)
        self.equity = self._calculate_equity(bid, ask)
        reward = self.equity - prev_equity
        self.history['equity'].append(self.equity)
        self.history['actions'].append(action)

        terminated = self.equity <= 0
        truncated = self.current_step >= self.total_steps - 1

        return self._get_observation(), reward, terminated, truncated, {
            'equity': self.equity,
            'actions': self.history['actions'],
            'equity_curve': self.history['equity']
        }

    def _trade(self, action, bid, ask):
        if action == 0:
            if self.position == 1:
                proceeds = self.shares * bid
                fee = proceeds * self.transaction_cost_pct
                self.cash += proceeds - fee
                self.shares = 0
                self.position = 0
            elif self.position == -1:
                cost = abs(self.shares) * ask
                fee = cost * self.transaction_cost_pct
                self.cash -= cost + fee
                self.shares = 0
                self.position = 0
        elif action == 1:
            if self.position == -1:
                self._trade(0, bid, ask)
            if self.position == 0 and ask > 0:
                size = self.cash / (ask * (1 + self.transaction_cost_pct))
                cost = size * ask
                fee = cost * self.transaction_cost_pct
                self.cash -= cost + fee
                self.shares = size
                self.position = 1
        elif action == 2:
            if self.position == 1:
                self._trade(0, bid, ask)
            if self.position == 0 and bid > 0:
                size = self.cash / bid
                proceeds = size * bid
                fee = proceeds * self.transaction_cost_pct
                self.cash += proceeds - fee
                self.shares = -size
                self.position = -1

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, Position: {self.position}, Cash: {self.cash:.2f}, Shares: {self.shares:.2f}")
