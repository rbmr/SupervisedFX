import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stockstats import StockDataFrame
import time

INITIAL_CAPITAL = 10000.0
TRANSACTION_COST_PCT = 0.001
LOOKBACK_WINDOW_SIZE = 30

RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

class ForexEnv(gym.Env):
    """
    Improved Forex trading environment for reinforcement learning.
    Actions: 0 - HOLD, 1 - LONG, 2 - SHORT, 3 - CASH/CLOSE
    """

    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, df, initial_capital=INITIAL_CAPITAL, transaction_cost_pct=TRANSACTION_COST_PCT,
                 lookback_window_size=LOOKBACK_WINDOW_SIZE, log_level=1, seed=None, debug_mode=False): # Added debug_mode
        super(ForexEnv, self).__init__()

        self.df = df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_window_size = lookback_window_size
        self.log_level = log_level # You can use this to control debug messages too
        self.seed = seed
        self.debug_mode = debug_mode # Store debug_mode

        if self.seed is not None:
            np.random.seed(self.seed)

        self.num_market_features = 6
        obs_size = self.lookback_window_size * self.num_market_features + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.trade_history = []
        if hasattr(self, 'principal_invested_current_pos'):
            del self.principal_invested_current_pos

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
        self.position = 0
        self.equity = self.initial_capital
        self.history = {'equity': [self.equity], 'actions': []}
        self.trade_history = []
        if hasattr(self, 'principal_invested_current_pos'):
            del self.principal_invested_current_pos
        if hasattr(self, 'entry_price_long'):
             del self.entry_price_long
        if hasattr(self, 'entry_price_short'):
             del self.entry_price_short
        if self.debug_mode:
            print(f"\n[RESET] Initial Capital: {self.initial_capital:.2f}, Equity: {self.equity:.2f}")
        return self._get_observation(), {}

    def _get_observation(self):
        i = self.current_step
        market_window = self.processed_df.iloc[i:i+self.lookback_window_size][[
            'log_ret_bid', 'log_ret_ask', 'norm_spread', 'norm_volume', 'norm_rsi', 'norm_macd_diff']].values.flatten()
        bid, ask = self._get_current_prices()
        # Equity for observation should be current, calculated if needed, but self.equity should be up-to-date
        # equity_for_obs = self._calculate_equity(bid, ask)
        cash_ratio = np.clip(self.cash / self.equity, -1e6, 1e6) if self.equity > 0 else 0.0
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
        if self.debug_mode:
            time.sleep(2)
        self.current_step += 1
        bid, ask = self._get_current_prices()

        prev_equity = self.equity
        prev_cash = self.cash # Store prev_cash for debugging
        prev_shares = self.shares # Store prev_shares for debugging
        prev_position = self.position
        prev_principal_invested = getattr(self, 'principal_invested_current_pos', 0.0)

        action_map = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE/CASH"}
        if self.debug_mode:
            print(f"\n--- Step {self.current_step} ---")
            print(f"[STEP_BEGIN] Action: {action} ({action_map.get(action, 'UNKNOWN')}), Bid: {bid:.4f}, Ask: {ask:.4f}")
            print(f"[PRE_TRADE_STATE] Equity: {prev_equity:.2f}, Cash: {prev_cash:.2f}, Shares: {prev_shares:.4f}, Position: {prev_position}, PrincipalInvested: {prev_principal_invested:.2f}")

        trade_occurred, principal_committed_to_newly_opened_position = self._trade(action, bid, ask)

        # Equity is updated based on the NEW state of cash and shares after _trade
        current_equity_calculated = self._calculate_equity(bid, ask)
        equity_diff_debug = current_equity_calculated - self.equity # Should be 0 if self.equity was updated after _calculate_equity
        self.equity = current_equity_calculated


        if self.debug_mode:
            print(f"[POST_TRADE_STATE] Trade Occurred: {trade_occurred}, PrincipalCommittedForNew: {principal_committed_to_newly_opened_position:.2f}")
            print(f"[POST_TRADE_STATE] Equity: {self.equity:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}, Position: {self.position}")
            if hasattr(self, 'principal_invested_current_pos'):
                print(f"[POST_TRADE_STATE] Current PrincipalInvested: {self.principal_invested_current_pos:.2f}")
            if abs(self.equity - prev_equity) > 1e-3 : # Print if equity changed significantly
                 print(f"[DEBUG] Equity change this step (before reward): {self.equity - prev_equity:.2f}")


        reward = 0.0
        reward_info = "N/A"

        if not trade_occurred and action != 0:
            reward = -0.1
            reward_info = f"Invalid action penalty ({action_map.get(action, 'UNKNOWN')} failed)"
        elif action == 0:
            if self.position != 0:
                reward = (self.equity - prev_equity) / max(prev_equity, 1e-9)
                if abs(reward) < 1e-7: reward = 0.0
                reward_info = f"Holding position {self.position}, Unrealized PnL ratio"
            else:
                reward = -0.0001
                reward_info = "Holding cash penalty"
        elif trade_occurred:
            if prev_position == 0 and self.position != 0:
                reward = 0.0
                if principal_committed_to_newly_opened_position > 0:
                    self.principal_invested_current_pos = principal_committed_to_newly_opened_position
                    reward_info = f"Opened new position {self.position}, Principal set: {self.principal_invested_current_pos:.2f}"
                else:
                    reward_info = f"Opened new position {self.position}, but principal_committed was 0 (ERROR?)"

            elif prev_position != 0 and self.position == 0:
                current_principal = getattr(self, 'principal_invested_current_pos', 0)
                if current_principal > 0:
                    realized_pnl_value = self.equity - current_principal
                    reward = realized_pnl_value / current_principal
                    reward_info = f"Closed position {prev_position}. Realized PnL: {realized_pnl_value:.2f} from Principal: {current_principal:.2f}. PnL Ratio for reward."
                    del self.principal_invested_current_pos
                else:
                    reward = (self.equity - prev_equity) / max(prev_equity, 1e-9)
                    reward_info = f"Closed position {prev_position} (Fallback reward - no principal found). Equity change ratio."
            elif prev_position != 0 and self.position != 0 and prev_position != self.position:
                reward = (self.equity - prev_equity) / max(prev_equity, 1e-9)
                reward_info = f"Switched position from {prev_position} to {self.position}. Equity change ratio."
                if principal_committed_to_newly_opened_position > 0:
                    self.principal_invested_current_pos = principal_committed_to_newly_opened_position
                    reward_info += f" New principal set: {self.principal_invested_current_pos:.2f}"
                else:
                     reward_info += " New principal committed was 0 (ERROR in switch open leg?)"


        if self.position == 0 and hasattr(self, 'principal_invested_current_pos'):
            if self.debug_mode:
                print(f"[DEBUG] Cleanup: In cash, deleting 'principal_invested_current_pos' which was {self.principal_invested_current_pos:.2f}")
            del self.principal_invested_current_pos
        
        # Sanity check for principal when in position
        if self.debug_mode and self.position != 0 and not hasattr(self, 'principal_invested_current_pos'):
            print(f"[DEBUG_WARNING] In position {self.position}, but 'principal_invested_current_pos' is NOT set!")


        self.history['equity'].append(self.equity)
        self.history['actions'].append(action)

        terminated = self.equity <= 0 or (self.cash < -1e-3 and self.position != -1)
        if self.position == -1 and self.cash < -self.initial_capital * 1.5 : # Reduced threshold for safety
            terminated = True
        truncated = self.current_step >= self.total_steps -1

        if self.debug_mode:
            print(f"[REWARD_CALC] Reward: {reward:.7f} ({reward_info})")
            print(f"[FINAL_STATE] Equity: {self.equity:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}, Position: {self.position}")
            if hasattr(self, 'principal_invested_current_pos'):
                 print(f"[FINAL_STATE] Current PrincipalInvested: {self.principal_invested_current_pos:.2f}")
            print(f"[TERMINATION] Terminated: {terminated}, Truncated: {truncated}")
            print(f"--- End Step {self.current_step} ---")


        info = {
            'equity': self.equity, 'cash': self.cash, 'shares': self.shares, 'position': self.position,
            'actions': self.history['actions'][-10:] if self.current_step > 10 else self.history['actions'],
            'equity_curve': self.history['equity'], 'reward': reward
        }
        if terminated or truncated:
            info['trade_history'] = self.trade_history

        return self._get_observation(), reward, terminated, truncated, info

    def _trade(self, action, bid_price, ask_price):
        trade_occurred = False
        principal_committed = 0.0
        action_map = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE/CASH"}

        if self.debug_mode:
            print(f"  [_trade ENTRY] Action: {action} ({action_map.get(action, 'UNKNOWN')}), Current Pos: {self.position}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}, Bid: {bid_price:.4f}, Ask: {ask_price:.4f}")

        if action == 0:
            if self.debug_mode: print("  [_trade] Action HOLD, no trade.")
            return False, 0.0

        # Action 1: Open LONG
        if action == 1:
            if self.position == 0:
                if self.cash > 1e-3 and ask_price > 0:
                    principal_committed = self.cash
                    if self.debug_mode: print(f"  [_trade OPEN_LONG_FROM_CASH] Attempting with cash: {principal_committed:.2f}")
                    cost_per_share_incl_commission = ask_price * (1 + self.transaction_cost_pct)
                    num_shares = 0
                    if cost_per_share_incl_commission > 0:
                        num_shares = self.cash / cost_per_share_incl_commission
                    else:
                        if self.debug_mode: print(f"  [_trade OPEN_LONG_FROM_CASH] Failed: cost_per_share_incl_commission is {cost_per_share_incl_commission:.4f}")
                        return False, 0.0
                    
                    actual_cost = num_shares * ask_price
                    commission = actual_cost * self.transaction_cost_pct
                    if self.debug_mode: print(f"  [_trade OPEN_LONG_FROM_CASH] Calculated: num_shares={num_shares:.4f}, actual_cost={actual_cost:.2f}, commission={commission:.2f}")

                    if self.cash >= actual_cost + commission:
                        self.cash -= (actual_cost + commission)
                        self.shares = num_shares
                        self.position = 1
                        self.entry_price_long = ask_price # Still useful for some debug/analysis if needed
                        self.trade_history.append((self.current_step, "OPEN_LONG", ask_price, self.shares, commission))
                        trade_occurred = True
                        if self.debug_mode: print(f"  [_trade OPEN_LONG_FROM_CASH] SUCCESS. New Cash: {self.cash:.2f}, Shares: {self.shares:.4f}, Pos: {self.position}")
                    else:
                        if self.debug_mode: print(f"  [_trade OPEN_LONG_FROM_CASH] Failed: Insufficient cash for cost+commission. Have {self.cash:.2f}, Need {actual_cost + commission:.2f}")
                        principal_committed = 0.0 # Failed to open
                else:
                    if self.debug_mode: print(f"  [_trade OPEN_LONG_FROM_CASH] Condition not met: Cash ({self.cash:.2f}) <= 1e-3 or Ask Price ({ask_price:.4f}) <= 0")
            elif self.position == -1: # Switch from SHORT to LONG
                if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Current cash: {self.cash:.2f}, shares: {self.shares:.4f}")
                # 1. Close SHORT
                cost_to_cover = abs(self.shares) * ask_price
                commission_close = cost_to_cover * self.transaction_cost_pct
                cash_before_close = self.cash
                self.cash -= (cost_to_cover + commission_close)
                closed_shares = abs(self.shares)
                # self.shares = 0 # Set shares to 0 after confirming transaction
                # self.position = 0 # Set position to 0 after confirming transaction
                trade_history_entry_close = (self.current_step, "CLOSE_SHORT_FOR_LONG", ask_price, closed_shares, commission_close)
                if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Closing Short: cost_to_cover={cost_to_cover:.2f}, commission_close={commission_close:.2f}. Cash change: {self.cash - cash_before_close:.2f}. New cash: {self.cash:.2f}")
                
                # Temporarily set state as if short is closed
                original_shares = self.shares
                self.shares = 0
                self.position = 0
                trade_occurred = True # At least the close part happened

                if hasattr(self, 'principal_invested_current_pos'): del self.principal_invested_current_pos
                if hasattr(self, 'entry_price_short'): del self.entry_price_short

                # 2. Open LONG
                if self.cash > 1e-3 and ask_price > 0:
                    principal_committed = self.cash # This is the cash available for the new long
                    if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Attempting Open Long leg with cash: {principal_committed:.2f}")
                    cost_per_share_incl_commission = ask_price * (1 + self.transaction_cost_pct)
                    num_shares_new_long = 0
                    if cost_per_share_incl_commission > 0:
                        num_shares_new_long = self.cash / cost_per_share_incl_commission
                    else:
                        if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Open Long leg Failed: cost_per_share_incl_commission is {cost_per_share_incl_commission:.4f}")
                        self.trade_history.append(trade_history_entry_close) # Log the close part
                        # principal_committed remains cash after close, but new position is not opened
                        # Since new position is not opened, it's better to return 0 for principal_committed_to_newly_opened_position
                        return True, 0.0


                    actual_cost_new_long = num_shares_new_long * ask_price
                    commission_open_new_long = actual_cost_new_long * self.transaction_cost_pct
                    if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Open Long leg calc: num_shares={num_shares_new_long:.4f}, actual_cost={actual_cost_new_long:.2f}, commission={commission_open_new_long:.2f}")

                    if self.cash >= actual_cost_new_long + commission_open_new_long:
                        self.cash -= (actual_cost_new_long + commission_open_new_long)
                        self.shares = num_shares_new_long
                        self.position = 1
                        self.entry_price_long = ask_price
                        self.trade_history.append(trade_history_entry_close)
                        self.trade_history.append((self.current_step, "OPEN_LONG_AFTER_SWITCH", ask_price, self.shares, commission_open_new_long))
                        if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Open Long leg SUCCESS. New Cash: {self.cash:.2f}, Shares: {self.shares:.4f}, Pos: {self.position}")
                    else: # Failed to open long after closing short
                        if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Open Long leg FAILED: Insufficient cash. Have {self.cash:.2f}, Need {actual_cost_new_long + commission_open_new_long:.2f}")
                        self.trade_history.append(trade_history_entry_close) # Log the close part
                        principal_committed = 0.0 # Open leg failed
                        # self.position is already 0, self.shares is 0
                else: # Not enough cash or bad price for open long leg
                    if self.debug_mode: print(f"  [_trade SWITCH_SHORT_TO_LONG] Open Long leg condition not met: Cash ({self.cash:.2f}) <= 1e-3 or Ask Price ({ask_price:.4f}) <= 0")
                    self.trade_history.append(trade_history_entry_close) # Log the close part
                    principal_committed = 0.0 # Open leg failed
            else: # Tried to LONG but already LONG or other invalid state
                if self.debug_mode: print(f"  [_trade] Action LONG ({action}) invalid for current position {self.position}.")
                return False, 0.0


        # Action 2: Open SHORT
        elif action == 2:
            if self.position == 0:
                if self.cash > 1e-3 and bid_price > 0: # Using self.cash as notional value for shorting
                    principal_committed = self.cash # This is the notional principal
                    if self.debug_mode: print(f"  [_trade OPEN_SHORT_FROM_CASH] Attempting with notional cash value: {principal_committed:.2f}")
                    num_shares = self.cash / bid_price # How many shares to short to match current cash's notional value
                    
                    proceeds = num_shares * bid_price # Should be equal to principal_committed here
                    commission = proceeds * self.transaction_cost_pct
                    if self.debug_mode: print(f"  [_trade OPEN_SHORT_FROM_CASH] Calculated: num_shares={num_shares:.4f}, proceeds={proceeds:.2f}, commission={commission:.2f}")

                    self.cash += (proceeds - commission)
                    self.shares = -num_shares
                    self.position = -1
                    self.entry_price_short = bid_price
                    self.trade_history.append((self.current_step, "OPEN_SHORT", bid_price, abs(self.shares), commission))
                    trade_occurred = True
                    if self.debug_mode: print(f"  [_trade OPEN_SHORT_FROM_CASH] SUCCESS. New Cash: {self.cash:.2f}, Shares: {self.shares:.4f}, Pos: {self.position}")
                else:
                    if self.debug_mode: print(f"  [_trade OPEN_SHORT_FROM_CASH] Condition not met: Cash ({self.cash:.2f}) <= 1e-3 or Bid Price ({bid_price:.4f}) <= 0")
            elif self.position == 1: # Switch from LONG to SHORT
                if self.debug_mode: print(f"  [_trade SWITCH_LONG_TO_SHORT] Current cash: {self.cash:.2f}, shares: {self.shares:.4f}")
                # 1. Close LONG
                proceeds_from_long = self.shares * bid_price
                commission_close = proceeds_from_long * self.transaction_cost_pct
                cash_before_close = self.cash
                self.cash += (proceeds_from_long - commission_close)
                closed_shares = self.shares
                # self.shares = 0 # Set after confirming
                # self.position = 0 # Set after confirming
                trade_history_entry_close = (self.current_step, "CLOSE_LONG_FOR_SHORT", bid_price, closed_shares, commission_close)
                if self.debug_mode: print(f"  [_trade SWITCH_LONG_TO_SHORT] Closing Long: proceeds={proceeds_from_long:.2f}, commission_close={commission_close:.2f}. Cash change: {self.cash - cash_before_close:.2f}. New cash: {self.cash:.2f}")

                original_shares = self.shares # For debug
                self.shares = 0
                self.position = 0
                trade_occurred = True # At least the close part happened

                if hasattr(self, 'principal_invested_current_pos'): del self.principal_invested_current_pos
                if hasattr(self, 'entry_price_long'): del self.entry_price_long

                # 2. Open SHORT
                if self.cash > 1e-3 and bid_price > 0:
                    principal_committed = self.cash # Cash available for new short (as notional value)
                    if self.debug_mode: print(f"  [_trade SWITCH_LONG_TO_SHORT] Attempting Open Short leg with notional cash: {principal_committed:.2f}")
                    num_shares_new_short = self.cash / bid_price
                    
                    proceeds_new_short = num_shares_new_short * bid_price
                    commission_open_new_short = proceeds_new_short * self.transaction_cost_pct
                    if self.debug_mode: print(f"  [_trade SWITCH_LONG_TO_SHORT] Open Short leg calc: num_shares={num_shares_new_short:.4f}, proceeds={proceeds_new_short:.2f}, commission={commission_open_new_short:.2f}")
                    
                    self.cash += (proceeds_new_short - commission_open_new_short)
                    self.shares = -num_shares_new_short
                    self.position = -1
                    self.entry_price_short = bid_price
                    self.trade_history.append(trade_history_entry_close)
                    self.trade_history.append((self.current_step, "OPEN_SHORT_AFTER_SWITCH", bid_price, abs(self.shares), commission_open_new_short))
                    if self.debug_mode: print(f"  [_trade SWITCH_LONG_TO_SHORT] Open Short leg SUCCESS. New Cash: {self.cash:.2f}, Shares: {self.shares:.4f}, Pos: {self.position}")
                else: # Failed to open short after closing long
                    if self.debug_mode: print(f"  [_trade SWITCH_LONG_TO_SHORT] Open Short leg FAILED: Condition not met. Cash ({self.cash:.2f}) or Bid Price ({bid_price:.4f}) invalid.")
                    self.trade_history.append(trade_history_entry_close) # Log the close part
                    principal_committed = 0.0 # Open leg failed
                    # self.position is already 0, self.shares is 0
            else: # Tried to SHORT but already SHORT or other invalid state
                 if self.debug_mode: print(f"  [_trade] Action SHORT ({action}) invalid for current position {self.position}.")
                 return False, 0.0


        # Action 3: Close current position
        elif action == 3 and self.position != 0:
            if self.debug_mode: print(f"  [_trade CLOSE_POSITION] Current Pos: {self.position}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}")
            if self.position == 1: # Close LONG
                proceeds = self.shares * bid_price
                commission = proceeds * self.transaction_cost_pct
                self.cash += (proceeds - commission)
                self.trade_history.append((self.current_step, "CLOSE_LONG", bid_price, self.shares, commission))
                if self.debug_mode: print(f"  [_trade CLOSE_LONG] Proceeds: {proceeds:.2f}, Commission: {commission:.2f}. New Cash: {self.cash:.2f}")
                if hasattr(self, 'entry_price_long'): del self.entry_price_long
            elif self.position == -1: # Close SHORT
                cost = abs(self.shares) * ask_price
                commission = cost * self.transaction_cost_pct
                self.cash -= (cost + commission)
                self.trade_history.append((self.current_step, "CLOSE_SHORT", ask_price, abs(self.shares), commission))
                if self.debug_mode: print(f"  [_trade CLOSE_SHORT] Cost to cover: {cost:.2f}, Commission: {commission:.2f}. New Cash: {self.cash:.2f}")
                if hasattr(self, 'entry_price_short'): del self.entry_price_short
            
            self.shares = 0
            self.position = 0
            trade_occurred = True
            principal_committed = 0.0 # No new position opened
            if self.debug_mode: print(f"  [_trade CLOSE_POSITION] Position closed. New Pos: {self.position}, Shares: {self.shares:.4f}")
        elif action == 3 and self.position == 0:
            if self.debug_mode: print(f"  [_trade CLOSE_POSITION] Attempted to close, but already in cash. No trade.")
            return False, 0.0


        if self.debug_mode:
            print(f"  [_trade EXIT] Returning: trade_occurred={trade_occurred}, principal_committed_for_new_open={principal_committed:.2f}")
        return trade_occurred, principal_committed


    def render(self, mode='human'):
        # Original render is fine, or you can enhance it.
        # The debug messages in step() will provide a lot of info if debug_mode is True.
        position_map = {0: "CASH", 1: "LONG", -1: "SHORT"}
        print(f"Render Step: {self.current_step}, Equity: {self.equity:.2f}, Position: {position_map.get(self.position, 'UNKNOWN')}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}")
        if hasattr(self, 'principal_invested_current_pos'):
            print(f"  Render Principal Invested: {self.principal_invested_current_pos:.2f}")