import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.scripts import clean_numpy, write_atomic_json


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

def calculate_streaks(win_conditions: np.ndarray, lose_conditions: np.ndarray):
    """Calculate winning and losing streaks using vectorized operations"""
    assert win_conditions.shape == lose_conditions.shape
    assert win_conditions.ndim == lose_conditions.ndim == 1

    # Create sequence of wins (1), losses (-1), and neutrals (0)
    sequence = np.zeros(len(win_conditions))
    sequence[win_conditions] = 1
    sequence[lose_conditions] = -1

    return get_max_streak(sequence, 1), get_max_streak(sequence, -1)

def drop_and_return_numpy(df: pd.DataFrame, cols: list[str]) -> list[np.ndarray]:
    arrays = []
    for col in cols:
        arr = df[col].to_numpy(dtype=df[col].dtype)
        arrays.append(arr)
        df.drop(columns=col, inplace=True)
    return arrays

def analyse_individual_run(results_file: Path, model_name: str):
    """
    Analyzes a data.csv file and outputs resulting files in the same directory.
    """
    if not results_file.is_file():
        raise FileNotFoundError(results_file)
    if results_file.suffix != ".csv":
        raise ValueError(f"results_file must have .csv extension, was {results_file.suffix}")
    output_dir = results_file.parent
    info_file = output_dir / "info.json"
    # Skip if the final info file already exists and the contents are readable.
    if info_file.exists():
        try:
            with open(info_file, "r") as f:
                _ = json.load(f)
            return
        except Exception:
            pass

    # Load results
    all_columns = pd.read_csv(results_file, nrows=0).columns.tolist()
    dtypes = {col: 'float32' for col in all_columns}
    dtypes["step"] = 'int32'
    dtypes["done"] = 'boolean'
    dtypes["info.market_data.date_gmt"] = 'float64'
    df = pd.read_csv(results_file, dtype=dtypes)

    # Analyze results
    logging.info(f"Analyzing {results_file}")

    # Ensure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # calculate correlation matrix of the dataframe
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Correlation Matrix for {model_name}")
    plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
    plt.tight_layout()
    plt.savefig(output_dir / f"correlation_matrix.png")
    plt.close()

    # Extract arrays that will be used
    np_columns = drop_and_return_numpy(df,[
        'info.market_data.close_bid', 'info.market_data.close_ask',
        'info.agent_data.equity_open', 'info.agent_data.equity_high',
        'info.agent_data.equity_low', 'info.agent_data.equity_close',
        'info.agent_data.action', 'info.market_data.date_gmt',
        'reward'
    ])
    close_bid, close_ask, equity_open, equity_high, equity_low, equity_close, actions, dates, rewards = np_columns
    del df # df is no longer necessary

    # Plot market data
    plt.figure(figsize=(12, 6))
    plt.plot(close_bid, label='Close Bid Prices')
    plt.plot(close_ask, label='Close Ask Prices')
    plt.title(f"Close Prices for {model_name}")
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(output_dir / f"market_data.png")
    plt.close()

    # Plot equity OHLC
    plt.figure(figsize=(12, 6))
    plt.plot(equity_open, label='Equity Open', color='blue')
    plt.plot(equity_high, label='Equity High', color='green')
    plt.plot(equity_low, label='Equity Low', color='red')
    plt.plot(equity_close, label='Equity Close', color='orange')
    plt.title(f"Equity OHLC for {model_name}")
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(output_dir / f"equity_ohlc.png")
    plt.close()

    # Calculate sharpe ratio on the close prices of equity.
    equity_returns = np.diff(equity_close) / equity_close[:-1]
    mean_return = np.mean(equity_returns)
    std_return = np.std(equity_returns, ddof=1)
    if std_return > 0:
        sharpe_ratio = mean_return / std_return
    else:
        sharpe_ratio = 0.0

    # Annualize sharpe ratio
    min_date = dates.min()
    max_date = dates.max()
    date_range_ns = max_date - min_date
    amount_years = date_range_ns / (365.25 * 24 * 60 * 60 * 1e9)

    if amount_years > 0:
        N = equity_returns.shape[0] / amount_years
        sharpe_ratio = sharpe_ratio * np.sqrt(N)

    # Vectorized drawdown calculation
    cummax_equity = np.maximum.accumulate(equity_close)
    drawdown = equity_close - cummax_equity
    max_drawdown = np.min(drawdown)
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown, label='Drawdown', color='purple')
    plt.title(f"Drawdown for {model_name}")
    plt.xlabel('Time Step')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.savefig(output_dir / f"drawdown.png")
    plt.close()

    # Plot histogram of actions taken, dynamic to the actual values in the dataset
    plt.figure(figsize=(12, 6))
    plt.hist(actions, bins=16)
    plt.title(f"Actions Histogram for {model_name}")
    plt.xlabel('Action Value')
    plt.ylabel('Count')
    plt.savefig(output_dir / f"actions_histogram.png")
    plt.close()

    # scatter plot for the actions per time step
    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(len(actions)), actions, alpha=0.5)
    plt.title(f"Actions Scatter Plot for {model_name}")
    plt.xlabel('Time Step')
    plt.ylabel('Action Value')
    plt.savefig(output_dir / f"actions_scatter.png")
    plt.close()
    
    # profit factor. Calculate the gross profit (all positive returns) and gross loss (all negative returns)
    positive_returns = equity_returns[equity_returns > 0]
    negative_returns = equity_returns[equity_returns < 0]
    gross_profit = np.sum(positive_returns)
    gross_loss = -np.sum(negative_returns)
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    # ##################################################################
    # #       TRADE DEFINITION LOGIC STARTS HERE                       #
    # ##################################################################

    # A trade is a continuous block of actions with the same sign (>0 or <0).
    # An action of 0 ends the current trade.
    # We use np.sign() to get a sequence of 1 (long), -1 (short), and 0 (flat).
    trade_types = np.sign(actions).astype(np.int8)

    # Find where the trade type changes. Pad with 0 to catch trades at start/end.
    trade_changes = np.diff(np.concatenate(([0], trade_types, [0])))
    trade_boundaries = np.where(trade_changes != 0)[0]

    # Each boundary is the start of a new state (long, short, or flat)
    group_starts = trade_boundaries[:-1]
    group_ends = trade_boundaries[1:]

    # Filter out the "flat" periods where the action was 0
    is_trade_group = trade_types[group_starts] != 0
    trade_starts = group_starts[is_trade_group]
    trade_ends = group_ends[is_trade_group]

    if len(trade_starts) > 0:
        num_trades = len(trade_starts)
        
        # Get the type of each trade (1 for long, -1 for short)
        trade_types_grouped = trade_types[trade_starts]
        
        # Calculate returns and durations for each trade
        # A trade's return is equity at the end of the period minus equity at the start
        trade_returns = equity_close[trade_ends - 1] - equity_open[trade_starts]
        trade_durations = (dates[trade_ends - 1] - dates[trade_starts]) / (1e9 * 60 * 60) # in hours

        winning_trades = trade_returns > 0
        losing_trades = trade_returns < 0
        max_winning_streak, max_losing_streak = calculate_streaks(winning_trades, losing_trades)

        # Trade statistics
        total_trades = num_trades
        long_trades_count = np.sum(trade_types_grouped == 1)
        short_trades_count = np.sum(trade_types_grouped == -1)
        total_trades_returns = np.sum(trade_returns)
        average_trade_return = np.mean(trade_returns)
        # Calculate percentage return based on the equity at the start of each trade
        average_trade_return_pct = np.mean(trade_returns / equity_open[trade_starts])
        average_trade_duration_hours = np.mean(trade_durations)
        total_winning_rate = np.sum(winning_trades) / num_trades if num_trades > 0 else 0

    else:
        # No trades case
        total_trades = 0
        long_trades_count = 0
        short_trades_count = 0
        total_trades_returns = 0.0
        average_trade_return = 0.0
        average_trade_return_pct = 0.0
        average_trade_duration_hours = 0.0
        max_winning_streak = 0
        max_losing_streak = 0
        total_winning_rate = 0.0

    # ##################################################################
    # #       TRADE DEFINITION LOGIC ENDS HERE                         #
    # ##################################################################

    # average spread of close_bid and close_ask and open_bid and open_ask
    spread_bid = close_ask - close_bid
    average_spread = np.mean(spread_bid)

    # The cost of the spread is paid for each trade that is opened.
    estimated_total_spread_cost = total_trades * average_spread
    
    # rewards
    total_rewards = rewards.sum() 
    average_rewards = rewards.mean()
    

    # Prepare results
    info = {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,

        # Trades Summary
        "total_trades": total_trades,
        "long_trades_count": long_trades_count,
        "short_trades_count": short_trades_count,
        "total_trades_returns": total_trades_returns,
        "average_trade_return": average_trade_return,
        "average_trade_return_pct": average_trade_return_pct,
        "average_trade_duration_hours": average_trade_duration_hours,
        "max_winning_streak": max_winning_streak,
        "max_losing_streak": max_losing_streak,
        "total_winning_rate": total_winning_rate,

        # Additional metrics
        "average_spread": average_spread,
        "estimated_total_spread_cost": estimated_total_spread_cost,

        "total_rewards": total_rewards,
        "average_rewards": average_rewards,
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

        ('Average Spread', info['average_spread']),
        ('Estimated Total Spread Cost', info['estimated_total_spread_cost']),

        ('Total Rewards', info['total_rewards']),
        ('Average Rewards', info['average_rewards']),
    ]
    row_labels, table_data = zip(*metrics)
    table_data = [[val] for val in table_data]

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

    plt.title(f"Analysis Results for {model_name}", fontsize=12, y=1.05, weight='bold')
    plt.subplots_adjust(top=0.8)  # Adjust top margin to fit title
    plt.savefig(output_dir / f"analysis_results_table.png")
    plt.close()

    # log results
    write_atomic_json(clean_numpy(info), info_file)

def analyse_finals(final_metrics: List[Dict[str, Any]], output_dir: Path, env_name: str) -> None:
    """
    Analyze the final results DataFrame and save the analysis to the output_dir.
    """
    # Ensure results_path exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # make a plot of the sharpe ratios
    sharpe_ratios = [metrics['sharpe_ratio'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sharpe_ratios)), sharpe_ratios, tick_label=[f"{i + 1}" for i in range(len(sharpe_ratios))])
    plt.title(f"Sharpe Ratios for {env_name}")
    plt.xlabel('Model')
    plt.ylabel('Sharpe Ratio')
    plt.savefig(output_dir / f"sharpe_ratios.png")
    plt.close()

    # make a plot of the max drawdowns
    max_drawdowns = [metrics['max_drawdown'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(max_drawdowns)), max_drawdowns, tick_label=[f"{i + 1}" for i in range(len(max_drawdowns))])
    plt.title(f"Max Drawdowns for {env_name}")
    plt.xlabel('Model')
    plt.ylabel('Max Drawdown')
    plt.savefig(output_dir / f"max_drawdowns.png")
    plt.close()

    # make a plot of the profit factors
    baseline = 1
    profit_factors = [metrics['profit_factor'] for metrics in final_metrics]
    profit_factors = [pf - baseline for pf in profit_factors]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(profit_factors)),
            profit_factors,
            tick_label=[f"{i + 1}" for i in range(len(profit_factors))])
    plt.title(f"Profit Factors for {env_name}")
    plt.xlabel('Model')
    plt.ylabel('Profit Factor')
    yticks = plt.yticks()[0] # Adjust y ticks back
    plt.yticks(yticks, [f"{y + baseline:.2f}" for y in yticks])
    plt.savefig(output_dir / f"profit_factors.png")
    plt.close()

    # make a plot of the total pnl
    total_trades_returns = [metrics['total_trades_returns'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(total_trades_returns)), total_trades_returns, tick_label=[f"{i + 1}" for i in range(len(total_trades_returns))])
    plt.title(f"Total Trades Returns for {env_name}")
    plt.xlabel('Model')
    plt.ylabel('Total Trades Returns')
    plt.savefig(output_dir / f"total_trades_returns.png")
    plt.close()