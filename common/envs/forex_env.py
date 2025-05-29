import logging
import copy


import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import NamedTuple, Optional, Callable, Dict, Any

from common.data import ForexCandleData
from common.feature.feature_engineer import FeatureEngineer
from common.feature.stepwise_feature_engineer import StepwiseFeatureEngineer
from common.constants import *
from common.scripts import *
from common.scripts import find_first_row_without_nan
from common.optimization import DataFrameToNumPyAccessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleOHLC(NamedTuple):
    """
    Class used for storing the OHLC data of the agent.
    """
    open: float
    high: float
    low: float
    close: float

class OHLCV(NamedTuple):
    open_bid: float
    open_ask: float
    high_bid: float
    high_ask: float
    low_bid: float
    low_ask: float
    close_bid: float
    close_ask: float
    volume: float
    date_gmt: pd.Timestamp

class GeneralForexEnv(gym.Env):

    def __init__(self, 
                 market_data_df: pd.DataFrame,
                 data_feature_engineer: FeatureEngineer,
                 agent_feature_engineer: StepwiseFeatureEngineer,
                 initial_capital=10000.0,
                 transaction_cost_pct=0.0,
                 amount_actions=1,
                 custom_reward_function: Optional[Callable[['GeneralForexEnv'], float]] = None
                 ):
        super(GeneralForexEnv, self).__init__()

        # Validate input
        if transaction_cost_pct < 0.0 or transaction_cost_pct > 1.0: 
            raise ValueError(f"transaction_cost_pct must be between 0.0 and 1.0, was {transaction_cost_pct}. For 0.1%, use 0.001.")
        if initial_capital <= 0.0:
            raise ValueError(f"initial_capital must be positive, was {initial_capital}.")
        
        # Check if market_data_df is a DataFrame and contains the required columns
        if not isinstance(market_data_df, pd.DataFrame):
            raise ValueError(f"market_data_df must be a pandas DataFrame, was {type(market_data_df)}.")
        required_columns = {"open_bid", "open_ask", "high_bid", "high_ask", "low_bid", "low_ask", "close_bid", "close_ask", "volume", "date_gmt"}
        if not required_columns.issubset(market_data_df.columns):
            raise ValueError(f"market_data_df must contain the following columns: {required_columns}, was {market_data_df.columns}.")
        
        if (amount_actions == 0):
            raise ValueError(f"amount_actions must be greater than 0. This is used to determine the number of actions for both the buy and sell between the values of 0 and 1. For example, if amount_actions = 1, then the actions are seel100, sell50, nothing, buy50, buy 100. \nIf amount_actions < 0, then the action space is continuous between 0 and 1.")
            
        self.snapshot_trace = []

        # Environment settings
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.agent_feature_engineer = agent_feature_engineer
        self.custom_reward_function = custom_reward_function

        # Market Data
        self.market_data_df = market_data_df
        self.market_feature_df = data_feature_engineer.run(self.market_data_df)
        # Remove leading rows with NaN values
        start_index = find_first_row_without_nan(self.market_data_df)
        start_index = max(start_index, find_first_row_without_nan(self.market_feature_df))
        self.market_data_df = self.market_data_df.iloc[start_index:]
        self.market_feature_df = self.market_feature_df.iloc[start_index:]
        # reset index
        self.market_data_df.reset_index(drop=True, inplace=True)
        self.market_feature_df.reset_index(drop=True, inplace=True)
        
        # Agent Data, init pd.DataFrame with columns ['cash', 'shares', 'equity_open', 'equity_high', 'equity_low', 'equity_close'] and set initial values for len(self.market_data_df) rows
        self.agent_data_df = pd.DataFrame(index=range(len(self.market_data_df)), columns=['cash', 'shares', 'equity_open', 'equity_high', 'equity_low', 'equity_close'])
        self.agent_data_df.iloc[0] = {
            'cash': self.initial_capital,
            'shares': 0.0,
            'equity_open': self.initial_capital,
            'equity_high': self.initial_capital,
            'equity_low': self.initial_capital,
            'equity_close': self.initial_capital
        }
        
        # Define action space
        self.amount_actions = amount_actions
        if (amount_actions < 0):
            logging.info(f"amount_actions is negative, using continuous action space between 0 and 1.")
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.using_discrete_action_space = False
        elif (amount_actions > 0):
            logging.info(f"amount_actions is positive, using discrete action space with {amount_actions} actions for buys + {amount_actions} for sells and 1 action for no_participation.")
            self.action_space = spaces.Discrete(2 * amount_actions + 1)
            self.using_discrete_action_space = True
            print(f"Action space: {self.action_space}")
        self.prev_action = None

        # Define observation space
        self.num_market_features = len(self.market_feature_df.columns)
        self.num_state_features = len(self.agent_feature_engineer.get_columns())
        self.observation_space_shape = self.num_market_features + self.num_state_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_shape,), dtype=np.float32)

        self.total_steps = len(self.market_data_df)

        # check market_data, market_feature_df and agent_data_df are the same length
        if len(self.market_data_df) != len(self.market_feature_df):
            raise ValueError(f"market_data_df and market_feature_df must be the same length, was {len(self.market_data_df)} and {len(self.market_feature_df)}.")
        if len(self.market_data_df) != len(self.agent_data_df):
            raise ValueError(f"market_data_df and agent_data_df must be the same length, was {len(self.market_data_df)} and {len(self.agent_data_df)}.")
        if len(self.market_feature_df) != len(self.agent_data_df):
            raise ValueError(f"market_feature_df and agent_data_df must be the same length, was {len(self.market_feature_df)} and {len(self.agent_data_df)}.")
        
        # Check if any of the columns in market_data_df contain NaN, and do the same for market_feature_df
        if self.market_data_df.isnull().values.any():
            # find index of first row with NaN values
            first_nan_index = self.market_data_df.index[self.market_data_df.isnull().any(axis=1)][0]

            raise ValueError(f"market_data_df contains NaN values. First NaN index: {first_nan_index}. Row: {self.market_data_df.iloc[first_nan_index]}.")
        if self.market_feature_df.isnull().values.any():
            # find index of first row with NaN values
            first_nan_index = self.market_feature_df.index[self.market_feature_df.isnull().any(axis=1)][0]
            raise ValueError(f"market_feature_df contains NaN values. First NaN index: {first_nan_index}. Row: {self.market_feature_df.iloc[first_nan_index]}.")
        
        # Initialize the fast DataFrama Accessors
        self.market_data_accessor = DataFrameToNumPyAccessor(self.market_data_df)
        self.market_feature_accessor = DataFrameToNumPyAccessor(self.market_feature_df)
        self.agent_data_accessor = DataFrameToNumPyAccessor(self.agent_data_df)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.snapshot_trace = []

        # Reset environment state
        self.current_step = 0

        # Agent Data, init pd.DataFrame with columns ['cash', 'shares', 'equity_open', 'equity_high', 'equity_low', 'equity_close'] and set initial values for len(self.market_data_df) rows
        self.agent_data_df = pd.DataFrame(index=range(len(self.market_data_df)), columns=['cash', 'shares', 'equity_open', 'equity_high', 'equity_low', 'equity_close'])
        self.agent_data_df.iloc[0] = {
            'cash': self.initial_capital,
            'shares': 0.0,
            'equity_open': self.initial_capital,
            'equity_high': self.initial_capital,
            'equity_low': self.initial_capital,
            'equity_close': self.initial_capital
        }
        self.agent_data_accessor = DataFrameToNumPyAccessor(self.agent_data_df)

        self.prev_action = None

        info = self._create_snapshot()

        return self._get_observation(), info

    def step(self, action):
        # Perform step
        self.current_step += 1

        #get current prices
        current_data = self._get_current_prices()

        # perform action at open prices
        current_cash = self.agent_data_accessor[self.current_step - 1, 'cash']
        current_shares = self.agent_data_accessor[self.current_step - 1, 'shares']
        new_cash, new_shares = self._execute_action(action, self.prev_action, current_data, current_cash, current_shares)
        new_equity_ohlc = self._calculate_equity(current_data, new_cash, new_shares)
        self.agent_data_accessor[self.current_step] = {
            'cash': new_cash,
            'shares': new_shares,
            'equity_open': new_equity_ohlc.open,
            'equity_high': new_equity_ohlc.high,
            'equity_low': new_equity_ohlc.low,
            'equity_close': new_equity_ohlc.close
        }

        # calculate reward
        reward = self._get_reward()

        # Determine done
        terminated = False
        truncated = False
        if self.agent_data_accessor[self.current_step, 'equity_close'] <= 0:
            terminated = True
            logging.info(f"Step {self.current_step}: Agent ruined. Equity: {self.agent_data_accessor[self.current_step, 'equity_close']}.")
        if self.current_step >= self.total_steps - 1:
            truncated = True
            logging.info(f"Step {self.current_step}: End of data reached.") 

        # Determine info dict 
        info = self._create_snapshot()
        # if terminated or truncated:
        #     # Episode is ending, put relevant final info here
        #     # Make a copy of the list to avoid issues if it's modified elsewhere later
        #     info['episode_snapshot_data'] = copy.deepcopy(self.snapshot_trace)
        #     # Add any other summary stats for the episode you might want

        self.prev_action = action
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _create_snapshot(self) -> Dict[str, Any]:
        """
        Creates a snapshot of the current state of the environment.
        """
    
        snapshot = {
            'step': self.current_step,
            'market_data': {
                'date_gmt': self.market_data_accessor[self.current_step,'date_gmt'],
                'open_bid': self.market_data_accessor[self.current_step,'open_bid'],
                'open_ask': self.market_data_accessor[self.current_step,'open_ask'],
                'high_bid': self.market_data_accessor[self.current_step,'high_bid'],
                'high_ask': self.market_data_accessor[self.current_step,'high_ask'],
                'low_bid': self.market_data_accessor[self.current_step,'low_bid'],
                'low_ask': self.market_data_accessor[self.current_step,'low_ask'],
                'close_bid': self.market_data_accessor[self.current_step,'close_bid'],
                'close_ask': self.market_data_accessor[self.current_step,'close_ask'],
                'volume': self.market_data_accessor[self.current_step,'volume'],
            },
            'agent_data': {
                'cash': self.agent_data_accessor[self.current_step,'cash'],
                'shares': self.agent_data_accessor[self.current_step,'shares'],
                'equity': {
                    'open': self.agent_data_accessor[self.current_step,'equity_open'],
                    'high': self.agent_data_accessor[self.current_step,'equity_high'],
                    'low': self.agent_data_accessor[self.current_step,'equity_low'],
                    'close': self.agent_data_accessor[self.current_step,'equity_close']
                }
            }
        }
        self.snapshot_trace.append(snapshot)

        return snapshot
    
    def _get_reward(self) -> float:
        """
        Calculates the reward based on the current equity.
        Uses a custom reward function if provided, otherwise defaults to equity change.
        """
        if self.custom_reward_function is not None:
            # Pass the entire environment instance to the custom reward function.
            # This gives the custom function flexible access to any environment state it might need.
            return self.custom_reward_function(self)
        else:
            # Default reward: change in equity from the previous step to the current step
            # This is calculated after the current step's data has been populated.
            return self.agent_data_accessor[self.current_step, 'equity_close'] - \
                   self.agent_data_accessor[self.current_step - 1, 'equity_close']

    def _get_current_prices(self) -> OHLCV:
        """
        Retrieves the current bid and ask prices from the dataframe.
        """
        data = OHLCV(
            open_bid=self.market_data_accessor[self.current_step, 'open_bid'],
            open_ask=self.market_data_accessor[self.current_step, 'open_ask'],
            high_bid=self.market_data_accessor[self.current_step, 'high_bid'],
            high_ask=self.market_data_accessor[self.current_step, 'high_ask'],
            low_bid=self.market_data_accessor[self.current_step, 'low_bid'],
            low_ask=self.market_data_accessor[self.current_step, 'low_ask'],
            close_bid=self.market_data_accessor[self.current_step, 'close_bid'],
            close_ask=self.market_data_accessor[self.current_step, 'close_ask'],
            volume=self.market_data_accessor[self.current_step, 'volume'],
            date_gmt=self.market_data_accessor[self.current_step, 'date_gmt']
        )
        return data

    def _calculate_equity(self, prices: OHLCV, cash: float, shares: float) -> float:
        """
        Calculates the equity based on current cash, shares and prices.
        """
        return SimpleOHLC(
            open=self._calculate_equity_from_prices(prices.open_bid, prices.open_ask, cash, shares),
            high=self._calculate_equity_from_prices(prices.high_bid, prices.high_ask, cash, shares),
            low=self._calculate_equity_from_prices(prices.low_bid, prices.low_ask, cash, shares),
            close=self._calculate_equity_from_prices(prices.close_bid, prices.close_ask, cash, shares)
        )
    
    def _calculate_equity_from_prices(self, bid_price: float, ask_price: float, cash: float, shares: float) -> float:
        """
        Calculates the equity based on current cash, shares and prices.
        """
        return cash + (shares * bid_price) if shares > 0 else cash - (abs(shares) * ask_price)

    def _get_observation(self): 
        """
        Returns the current observation of the environment.
        The observation is a combination of market features and state features.
        """

        # Get market features
        market_features = self.market_feature_accessor[self.current_step]

        # Get state features
        state_features_dict = self.agent_feature_engineer.run(self.agent_data_accessor, self.current_step)
        state_features = np.array([
            state_features_dict[col] for col in self.agent_feature_engineer.get_columns()
        ])

        # Combine both features
        observation = np.concatenate((market_features, state_features), axis=0)
        
        return observation
    
    def _execute_action(self, raw_action: float, prev_raw_action: float, current_data: OHLCV, current_cash: float, current_shares: float) -> tuple[float, float]:
        """
        Determines the target position based on the agent's action (percentage of equity)
        and executes trades by calling buy or sell instrument methods.
        """
        # 1. Jitter Mitigation: If raw action hasn't changed significantly, do nothing.
        if prev_raw_action is not None and abs(raw_action - prev_raw_action) < 1e-5:
            # logging.debug(f"Step {self.current_step}: Raw action {raw_action} close to previous {prev_raw_action}. No trade due to jitter mitigation.")
            return current_cash, current_shares

        # 2. Determine target percentage from raw_action
        target_percentage = self._get_percentage_from_action(raw_action)

        # 3. Calculate current equity (mark-to-market)
        #    Equity calculation uses bid for valuing long positions, ask for valuing short positions (cost to close)
        current_equity = self._calculate_equity_from_prices(current_data.open_bid, current_data.open_ask, current_cash, current_shares)
        
        # Ensure equity is not negative for target calculation, as it might lead to reversed logic if agent is bankrupt
        # Although termination on equity <= 0 should prevent this during active trading.
        if current_equity <= 0: 
            # logging.warning(f"Step {self.current_step}: Current equity is zero or negative ({current_equity:.2f}). No further trades possible to alter position based on percentage.")
            return current_cash, current_shares


        # 4. Determine target position value in currency
        target_position_value = target_percentage * current_equity
        
        # 5. Determine target number of shares based on target value
        target_num_shares = 0.0
        if abs(target_percentage) < 1e-6: # Target is to be flat
            target_num_shares = 0.0
        elif target_percentage > 0: # Target is LONG
            if current_data.open_ask > 1e-6: # Ensure price is valid for calculation
                target_num_shares = target_position_value / current_data.open_ask
            else: # Invalid price, cannot determine target shares, maintain current
                # logging.warning(f"Step {self.current_step}: Invalid ask price ({current_data.open_ask}) for calculating long target shares.")
                target_num_shares = current_shares 
        else: # Target is SHORT (target_percentage < 0)
            if current_data.open_bid > 1e-6: # Ensure price is valid
                # target_position_value is negative, target_num_shares will be negative
                target_num_shares = target_position_value / current_data.open_bid 
            else: # Invalid price, cannot determine target shares, maintain current
                # logging.warning(f"Step {self.current_step}: Invalid bid price ({current_data.open_bid}) for calculating short target shares.")
                target_num_shares = current_shares

        # 6. Determine the change in shares needed
        shares_to_trade = target_num_shares - current_shares

        # 7. Dispatch to buy or sell methods
        new_cash, new_shares = current_cash, current_shares # Initialize with current values

        if abs(shares_to_trade) < 1e-6: # Effectively no change needed in share count
            # logging.debug(f"Step {self.current_step}: Target shares effectively same as current. No trade execution needed.")
            return current_cash, current_shares

        if shares_to_trade > 0: # Need to buy
            new_cash, new_shares = self._buy_instrument(
                shares_to_buy_abs=shares_to_trade, 
                price=current_data.open_ask, 
                current_cash=current_cash, 
                current_shares=current_shares
            )
        elif shares_to_trade < 0: # Need to sell
            new_cash, new_shares = self._sell_instrument(
                shares_to_sell_abs=abs(shares_to_trade), 
                price=current_data.open_bid, 
                current_cash=current_cash, 
                current_shares=current_shares
            )
            
        return new_cash, new_shares
    
    def _get_percentage_from_action(self, action: float) -> float:
        if self.using_discrete_action_space:
            action = (action - self.amount_actions) / self.amount_actions
            action = np.clip(action, -1.0, 1.0)

        return action
    
    def _buy_instrument(self, shares_to_buy_abs: float, price: float, current_cash: float, current_shares: float) -> tuple[float, float]:
        """
        Executes a buy order for a specified absolute number of shares.

        Args:
            shares_to_buy_abs: The absolute number of shares to buy (must be positive).
            price: The ask price at which to buy.
            current_cash: The agent's current cash.
            current_shares: The agent's current share holdings (can be negative for short positions).

        Returns:
            A tuple (updated_cash, updated_shares).
        """
        if shares_to_buy_abs < 1e-6:  # Effectively zero shares, no action needed
            return current_cash, current_shares
        if price <= 1e-6:  # Invalid price
            logging.warning(f"Step {self.current_step}: Attempted to buy with invalid price: {price}")
            return current_cash, current_shares # No transaction

        # Calculate cost including commission
        cost_per_share_incl_commission = price * (1 + self.transaction_cost_pct)

        affordable_shares = 0.0
        if cost_per_share_incl_commission > 1e-6: # Avoid division by zero and ensure cost is positive
            # Agent can't spend more cash than they have.
            # If current_cash is negative, affordable_shares will be <= 0.
            affordable_shares = max(0.0, current_cash / cost_per_share_incl_commission)
        
        actual_shares_bought = min(shares_to_buy_abs, affordable_shares)

        if actual_shares_bought > 1e-6: # If a transaction is actually happening
            cost_before_commission = actual_shares_bought * price
            total_commission = cost_before_commission * self.transaction_cost_pct
            
            updated_cash = current_cash - (cost_before_commission + total_commission)
            updated_shares = current_shares + actual_shares_bought
            # logging.info(f"Step {self.current_step}: Bought {actual_shares_bought:.4f} shares at {price:.5f}. Cash: {updated_cash:.2f}, Shares: {updated_shares:.4f}")
            return updated_cash, updated_shares
        else:
            # logging.info(f"Step {self.current_step}: Buy order for {shares_to_buy_abs:.4f} shares not executed (e.g. insufficient funds or zero amount).")
            return current_cash, current_shares # No transaction occurred
        
    def _sell_instrument(self, shares_to_sell_abs: float, price: float, current_cash: float, current_shares: float) -> tuple[float, float]:
        """
        Executes a sell order, potentially closing a long position and/or initiating/increasing a short position.
        Shorting is limited by a margin rule: total absolute short position value * 2 <= available cash.

        Args:
            shares_to_sell_abs: The absolute number of shares to sell (must be positive).
            price: The bid price at which to sell.
            current_cash: The agent's current cash.
            current_shares: The agent's current share holdings.

        Returns:
            A tuple (updated_cash, updated_shares).
        """
        if shares_to_sell_abs < 1e-6:  # Effectively zero shares
            return current_cash, current_shares
        if price <= 1e-6:  # Invalid price
            logging.warning(f"Step {self.current_step}: Attempted to sell with invalid price: {price}")
            return current_cash, current_shares

        # Initialize updated values with current state
        updated_cash = current_cash
        updated_shares = current_shares

        # Part 1: Handle closing of any existing long position
        shares_sold_from_long = 0.0
        if current_shares > 0: # If currently long
            shares_sold_from_long = min(shares_to_sell_abs, current_shares)
            
            if shares_sold_from_long > 1e-6:
                proceeds = shares_sold_from_long * price
                commission = proceeds * self.transaction_cost_pct
                updated_cash += (proceeds - commission)
                updated_shares -= shares_sold_from_long
                # logging.info(f"Step {self.current_step}: Closed long by selling {shares_sold_from_long:.4f} shares at {price:.5f}.")

        # Part 2: Handle shares intended for shorting (initiating or increasing short)
        # These are the shares remaining from the original sell order after closing any long position.
        shares_to_potentially_short = shares_to_sell_abs - shares_sold_from_long

        if shares_to_potentially_short > 1e-6:
            # Apply margin rule: max_total_abs_short_shares = available_cash / (2 * price)
            # available_cash for margin is updated_cash (cash after closing any long positions).
            
            max_total_abs_short_position_allowed = 0.0
            if price > 1e-6: # Denominator 2*price must be positive
                # Ensure updated_cash is not negative; if it is, no shorting capacity.
                max_total_abs_short_position_allowed = max(0.0, updated_cash / (2 * price))

            # current_abs_short_position is abs(updated_shares) because updated_shares is the state *after* closing longs.
            # At this point, updated_shares <= 0 if we are considering shorting.
            current_abs_short_portion = abs(updated_shares) # Should be 0 if we just closed a long, or abs of existing short.

            # How many *more* shares can we short without violating the total limit?
            allowable_additional_short_shares = max(0.0, max_total_abs_short_position_allowed - current_abs_short_portion)
            
            actual_shares_shorted = min(shares_to_potentially_short, allowable_additional_short_shares)

            if actual_shares_shorted > 1e-6:
                proceeds = actual_shares_shorted * price
                commission = proceeds * self.transaction_cost_pct
                updated_cash += (proceeds - commission) # Cash increases from short sale
                updated_shares -= actual_shares_shorted   # Makes share count more negative
                # logging.info(f"Step {self.current_step}: Shorted {actual_shares_shorted:.4f} shares at {price:.5f}. Margin limit applied.")
            # else:
                # logging.info(f"Step {self.current_step}: Short order for {shares_to_potentially_short:.4f} shares not executed or reduced due to margin limits.")
                
        return updated_cash, updated_shares
