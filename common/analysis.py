from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_streaks(win_conditions: np.ndarray, lose_conditions: np.ndarray):
    """Calculate winning and losing streaks using vectorized operations"""
    assert win_conditions.shape == lose_conditions.shape
    assert win_conditions.ndim == lose_conditions.ndim == 1

    # Create sequence of wins (1), losses (-1), and neutrals (0)
    sequence = np.zeros(len(win_conditions))
    sequence[win_conditions] = 1
    sequence[lose_conditions] = -1

    # Find streak lengths
    def get_max_streak(seq, target_value):
        if len(seq) == 0:
            return 0

        # Create groups of consecutive values
        changes = np.diff(np.concatenate(([0], seq, [0])) != target_value)
        start_indices = np.where(changes)[0][::2]
        end_indices = np.where(changes)[0][1::2]

        if len(start_indices) == 0:
            return 0

        streak_lengths = end_indices - start_indices
        return np.max(streak_lengths) if len(streak_lengths) > 0 else 0

    return get_max_streak(sequence, 1), get_max_streak(sequence, -1)

def calculate_trade_returns(open_signals: np.ndarray, close_signals: np.ndarray, prices: np.ndarray, trade_type: str='long'):
    """Vectorized calculation of trade returns"""
    assert trade_type in ('long', 'short')
    assert open_signals.shape == close_signals.shape == prices.shape
    assert open_signals.ndim == close_signals.ndim == prices.ndim == 1

    returns = np.zeros(len(prices))

    # Get indices where trades open and close
    open_indices = np.where(open_signals)[0]
    close_indices = np.where(close_signals)[0]

    # Match open and close signals
    for open_idx in open_indices:
        # Find the next close after this open
        future_closes = close_indices[close_indices > open_idx]
        if len(future_closes) > 0:
            close_idx = future_closes[0]
            if trade_type == 'long':
                returns[close_idx] = prices[close_idx] - prices[open_idx]
            else:  # short
                returns[close_idx] = prices[open_idx] - prices[close_idx]

    return returns

def analyse_individual_run(df: pd.DataFrame, results_path: Path, name: str) -> Dict[str, Any]:
    """
    Analyze the results DataFrame and save the analysis to the results_path.
    """
    # Ensure results_path exists
    results_path.mkdir(parents=True, exist_ok=True)

    # calculate correlation matrix of the dataframe
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Correlation Matrix for {name}")
    plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
    plt.tight_layout()
    plt.savefig(results_path / f"correlation_matrix.png")
    plt.close()

    # Pre-compute commonly used arrays (avoid repeated DataFrame access)
    data = df[['info.market_data.close_bid', 'info.market_data.close_ask',
               'info.agent_data.equity_open', 'info.agent_data.equity_high',
               'info.agent_data.equity_low', 'info.agent_data.equity_close',
               'info.agent_data.action']].values

    close_bid, close_ask, equity_open, equity_high, equity_low, equity_close, actions = data.T

    # Plot market data
    plt.figure(figsize=(12, 6))
    plt.plot(close_bid, label='Close Bid Prices')
    plt.plot(close_ask, label='Close Ask Prices')
    plt.title(f"Close Prices for {name}")
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(results_path / f"market_data.png")
    plt.close()

    # Plot equity OHLC
    plt.figure(figsize=(12, 6))
    plt.plot(equity_open, label='Equity Open', color='blue')
    plt.plot(equity_high, label='Equity High', color='green')
    plt.plot(equity_low, label='Equity Low', color='red')
    plt.plot(equity_close, label='Equity Close', color='orange')
    plt.title(f"Equity OHLC for {name}")
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(results_path / f"equity_ohlc.png")
    plt.close()

    # calculate sharpe ratio on the close prices of equity.
    # Then scale it to be annualized.
    # Base this annualization factor based on the number of steps in a year.
    # The timeframe can be anything, and no assumption is made about the time between steps.
    equity_returns = np.diff(equity_close) / equity_close[:-1]
    mean_return = np.mean(equity_returns)
    std_return = np.std(equity_returns, ddof=1)
    if std_return > 0:
        sharpe_ratio = mean_return / std_return
    else:
        sharpe_ratio = 0.0

    # Annualize the Sharpe ratio assuming 252 trading days in a year
    min_date = df['info.market_data.date_gmt'].min()
    max_date = df['info.market_data.date_gmt'].max()
    date_range = pd.to_datetime(max_date) - pd.to_datetime(min_date)
    amount_years = date_range.days / 365.25

    if amount_years > 0:
        N = equity_returns.shape[0] / amount_years
        sharpe_ratio = sharpe_ratio * np.sqrt(N)

    # Vectorized drawdown calculation
    cummax_equity = np.maximum.accumulate(equity_close)
    drawdown = equity_close - cummax_equity
    max_drawdown = np.min(drawdown)

    # Plot drawdown
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown, label='Drawdown', color='purple')
    plt.title(f"Drawdown for {name}")
    plt.xlabel('Time Step')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.savefig(results_path / f"drawdown.png")
    plt.close()

    # plot histogram of actions taken, dynamic to the actual values in the dataset
    unique_actions, counts = np.unique(actions, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique_actions, counts, width=0.1)
    plt.title(f"Actions Histogram for {name}")
    plt.xlabel('Action Value')
    plt.ylabel('Count')
    plt.xticks(unique_actions)
    plt.savefig(results_path / f"actions_histogram.png")
    plt.close()

    # profit factor. Calulate the gross profit (all positive returns) and gross loss (all negative returns)
    positive_returns = equity_returns[equity_returns > 0]
    negative_returns = equity_returns[equity_returns < 0]
    gross_profit = np.sum(positive_returns)
    gross_loss = -np.sum(negative_returns)
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')


    trades_df = df[['info.agent_data.action', 'info.market_data.date_gmt', 'info.agent_data.equity_open','info.agent_data.equity_close']].copy()
    trades_df['actions_prev'] = trades_df['info.agent_data.action'].shift(1).fillna(0)
    trades_df['trade_type'] = np.where(trades_df['info.agent_data.action'] > 0, 'long',
                                       np.where(trades_df['info.agent_data.action'] < 0, 'short', 'none'))
    # remove none trades
    trades_df = trades_df[trades_df['trade_type'] != 'none'].reset_index(drop=True)

    # group adjacent trades of the same type, and calculate beginning data, end data, begin equity, end equity
    trades_df['group'] = (trades_df['trade_type'] != trades_df['trade_type'].shift()).cumsum()
    trades_df = trades_df.groupby(['group', 'trade_type']).agg({
        'info.market_data.date_gmt': ['first', 'last'],
        'info.agent_data.equity_open': 'first',
        'info.agent_data.equity_close': 'last',
        'info.agent_data.action': ['first', 'last']
    }).reset_index()
    
    new_column_names = []
    for col in trades_df.columns.values:
        if isinstance(col, tuple):
            new_column_names.append('_'.join(col).strip('_'))
        else:
            new_column_names.append(str(col)) # Keep original name for non-tuple columns
    trades_df.columns = new_column_names

    trades_df.rename(columns={
        'info.market_data.date_gmt_first': 'trade_start_date',
        'info.market_data.date_gmt_last': 'trade_end_date',
        'info.agent_data.equity_open_first': 'trade_begin_equity',
        'info.agent_data.equity_close_last': 'trade_end_equity',
        'info.agent_data.action_first': 'trade_begin_action',
        'info.agent_data.action_last': 'trade_end_action'
    }, inplace=True)

    # Calculate trade returns
    trades_df['trade_return'] = trades_df['trade_end_equity'] - trades_df['trade_begin_equity']
    trades_df['trade_return_pct'] = trades_df['trade_return'] / trades_df['trade_begin_equity']
    trades_df['trade_duration_hours'] = (
        pd.to_datetime(trades_df['trade_end_date']) - pd.to_datetime(trades_df['trade_start_date'])
    ).dt.total_seconds() / 3600  # Convert to hours


    # calculate streaks of winning and losing trades
    trades_df['trade_winning'] = trades_df['trade_return'] > 0
    trades_df['trade_losing'] = trades_df['trade_return'] < 0
    trades_df['trade_winning_streak'] = trades_df['trade_winning'].astype(int).groupby((trades_df['trade_winning'] != trades_df['trade_winning'].shift()).cumsum()).cumsum()
    trades_df['trade_losing_streak'] = trades_df['trade_losing'].astype(int).groupby((trades_df['trade_losing'] != trades_df['trade_losing'].shift()).cumsum()).cumsum()    


    # Prepare results
    info = {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,

        # Trades Summary
        "total_trades": trades_df.shape[0],
        "long_trades_count": trades_df[trades_df['trade_type'] == 'long'].shape[0],
        "short_trades_count": trades_df[trades_df['trade_type'] == 'short'].shape[0],
        "total_trades_returns": trades_df['trade_return'].sum(),
        "average_trade_return": trades_df['trade_return'].mean(),
        "average_trade_return_pct": trades_df['trade_return_pct'].mean(),
        "average_trade_duration_hours": trades_df['trade_duration_hours'].mean(),
        "max_winning_streak": trades_df['trade_winning_streak'].max(),
        "max_losing_streak": trades_df['trade_losing_streak'].max(),
        "total_winning_rate":  trades_df[trades_df['trade_winning']].shape[0] / trades_df.shape[0] if trades_df.shape[0] > 0 else 0,
    }

    # Create results table
    metrics = [
        ('Sharpe Ratio', sharpe_ratio),
        ('Max Drawdown', max_drawdown),
        ('Profit Factor', profit_factor),

        ('Total Trades', info['total_trades']),
        ('Long Trades Count', info['long_trades_count']),
        ('Short Trades Count', info['short_trades_count']),
        ('Total Trades Returns', info['total_trades_returns']),
        ('Average Trade Return', info['average_trade_return']),
        ('Average Trade Return (%)', info['average_trade_return_pct']),
        ('Average Trade Duration (hours)', info['average_trade_duration_hours']),
        ('Max Winning Streak', info['max_winning_streak']),
        ('Max Losing Streak', info['max_losing_streak']),
        ('Total Winning Rate (%)', info['total_winning_rate'] * 100),
    ]
    row_labels, table_data = zip(*metrics)
    table_data = [[val] for val in table_data]

    # Create and save table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=['Metric Value'],
                    rowLabels=row_labels, cellLoc='center', loc='center',
                    colWidths=[0.5])
    table.scale(1.2, 1.2)

    cells = table.get_celld()
    for (row_idx, col_idx), cell in cells.items():
        if row_idx == 0 and col_idx >= 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('skyblue')
        elif col_idx == -1 and row_idx >= 0:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('lightgrey')

    plt.title(f"Analysis Results for {name}", fontsize=12, y=1.05, weight='bold')
    plt.subplots_adjust(top=0.8)  # Adjust top margin to fit title
    plt.savefig(results_path / f"analysis_results_table.png")
    plt.close()

    return info


def analyse_finals(final_metrics: List[Dict[str, Any]], results_path: Path, name: str) -> None:
    """
    Analyze the final results DataFrame and save the analysis to the results_path.
    """
    # Ensure results_path exists
    results_path.mkdir(parents=True, exist_ok=True)

    # make a plot of the sharpe ratios
    sharpe_ratios = [metrics['sharpe_ratio'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sharpe_ratios)), sharpe_ratios, tick_label=[f"{i + 1}" for i in range(len(sharpe_ratios))])
    plt.title(f"Sharpe Ratios for {name}")
    plt.xlabel('Model')
    plt.ylabel('Sharpe Ratio')
    plt.savefig(results_path / f"sharpe_ratios.png")
    plt.close()

    # make a plot of the max drawdowns
    max_drawdowns = [metrics['max_drawdown'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(max_drawdowns)), max_drawdowns, tick_label=[f"{i + 1}" for i in range(len(max_drawdowns))])
    plt.title(f"Max Drawdowns for {name}")
    plt.xlabel('Model')
    plt.ylabel('Max Drawdown')
    plt.savefig(results_path / f"max_drawdowns.png")
    plt.close()

    # make a plot of the profit factors
    baseline = 1
    profit_factors = [metrics['profit_factor'] for metrics in final_metrics]
    profit_factors = [pf - baseline for pf in profit_factors]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(profit_factors)),
            profit_factors,
            tick_label=[f"{i + 1}" for i in range(len(profit_factors))])
    plt.title(f"Profit Factors for {name}")
    plt.xlabel('Model')
    plt.ylabel('Profit Factor')
    yticks = plt.yticks()[0] # Adjust y ticks back
    plt.yticks(yticks, [f"{y + baseline:.2f}" for y in yticks])
    plt.savefig(results_path / f"profit_factors.png")
    plt.close()

    # make a plot of the total pnl
    total_trades_returns = [metrics['total_trades_returns'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(total_trades_returns)), total_trades_returns, tick_label=[f"{i + 1}" for i in range(len(total_trades_returns))])
    plt.title(f"Total Trades Returns for {name}")
    plt.xlabel('Model')
    plt.ylabel('Total Trades Returns')
    plt.savefig(results_path / f"total_trades_returns.png")
    plt.close()