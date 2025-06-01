from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

def analyse_individual_run(df: pd.DataFrame, results_path: Path, name: str) -> Dict[str, Any]:
    """
    Analyze the results DataFrame and save the analysis to the results_path.
    """
    # Ensure results_path exists
    results_path.mkdir(parents=True, exist_ok=True)

    # plot close prices of market data
    close_bid_prices = df['info.market_data.close_bid'].values
    close_ask_prices = df['info.market_data.close_ask'].values
    plt.figure(figsize=(12, 6))
    plt.plot(close_bid_prices, label='Close Bid Prices')
    plt.plot(close_ask_prices, label='Close Ask Prices')
    plt.title(f"Close Prices for {name}")
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(results_path / f"market_data.png")
    plt.close()

    # plote open high low close of equity
    equity_open = df['info.agent_data.equity_open'].values
    equity_high = df['info.agent_data.equity_high'].values
    equity_low = df['info.agent_data.equity_low'].values
    equity_close = df['info.agent_data.equity_close'].values
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
    # Then scale it to be annulaized.
    # Base this annulaization factor based on the number of steps in a year.
    # The timefram can be anything, and no assumption is made about the time between steps.
    equity_returns = df['info.agent_data.equity_close'].pct_change().dropna()
    mean_return = equity_returns.mean()
    std_return = equity_returns.std()
    if std_return == 0:
        sharpe_ratio = 0.0  # Avoid division by zero
    else:
        sharpe_ratio = mean_return / std_return
    # Annualize the Sharpe ratio assuming 252 trading days in a year
    min_date = df['info.market_data.date_gmt'].min()
    max_date = df['info.market_data.date_gmt'].max()
    date_range = pd.to_datetime(max_date) - pd.to_datetime(min_date)
    amount_years = date_range.days / 365.25  # Use 365.25 to account for leap years
    if amount_years == 0:
        sharpe_ratio = 0.0  # Avoid division by zero
    else:
        N = equity_returns.shape[0] / amount_years
        sharpe_ratio = sharpe_ratio * (N ** 0.5)  # Scale Sharpe ratio by sqrt(N)

    # Drawdown plot
    df['drawdown'] = df['info.agent_data.equity_close'] - df['info.agent_data.equity_close'].cummax()
    plt.figure(figsize=(12, 6))
    plt.plot(df['drawdown'], label='Drawdown', color='purple')
    plt.title(f"Drawdown for {name}")
    plt.xlabel('Time Step')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.savefig(results_path / f"drawdown.png")
    plt.close()
    max_drawdown = df['drawdown'].min()  # Maximum drawdown is the minimum value of the drawdown series

    # profit factor. Calulate the gross profit (all positive returns) and gross loss (all negative returns)
    df['returns'] = df['info.agent_data.equity_close'].pct_change().fillna(0)
    gross_profit = df[df['returns'] > 0]['returns'].sum()
    gross_loss = -df[df['returns'] < 0]['returns'].sum()  # Use negative returns for loss
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')  # Avoid division by zero

    # a long trade is opened when the agent_data.action goes from <= 0 to > 0
    df['open_long'] = (df['info.agent_data.action'] > 0) & (df['info.agent_data.action'].shift(1) <= 0)
    df['close_long'] = (df['info.agent_data.action'] <= 0) & (df['info.agent_data.action'].shift(1) > 0)
    # a short trade is opened when the agent_data.action goes from >= 0 to < 0
    df['open_short'] = (df['info.agent_data.action'] < 0) & (df['info.agent_data.action'].shift(1) >= 0)
    df['close_short'] = (df['info.agent_data.action'] >= 0) & (df['info.agent_data.action'].shift(1) < 0)

    long_trades_count = df['open_long'].sum()
    short_trades_count = df['open_short'].sum()
    total_trades_count = long_trades_count + short_trades_count

    # for close_long, determine the returns since the last open_long
    df['long_returns'] = 0.0
    last_open_long_index = None
    for i in range(len(df)):
        if df['open_long'].iloc[i]:
            last_open_long_index = i
        if last_open_long_index is not None and df['close_long'].iloc[i]:
            df.at[i, 'long_returns'] = df['info.agent_data.equity_close'].iloc[i] - df['info.agent_data.equity_close'].iloc[last_open_long_index]
    long_trades_returns = df['long_returns'].sum()
    # for close_short, determine the returns since the last open_short
    df['short_returns'] = 0.0
    last_open_short_index = None
    for i in range(len(df)):
        if df['open_short'].iloc[i]:
            last_open_short_index = i
        if last_open_short_index is not None and df['close_short'].iloc[i]:
            df.at[i, 'short_returns'] = df['info.agent_data.equity_close'].iloc[last_open_short_index] - df['info.agent_data.equity_close'].iloc[i]
    short_trades_returns = df['short_returns'].sum()
    # Calculate total returns from trades
    total_trades_returns = long_trades_returns + short_trades_returns

    # long winning trades
    long_winning_trades = df[df['close_long'] & (df['long_returns'] > 0)].shape[0]
    long_winning_ratio = long_winning_trades / long_trades_count if long_trades_count > 0 else 0.0

    # short winning trades
    short_winning_trades = df[df['close_short'] & (df['short_returns'] > 0)].shape[0]
    short_winning_ratio = short_winning_trades / short_trades_count if short_trades_count > 0 else 0.0

    # win rate
    total_winning_trades = long_winning_trades + short_winning_trades
    total_winning_ratio = total_winning_trades / total_trades_count if total_trades_count > 0 else 0.0

    # longest winning streak, defined by close_long or close_short with positive returns
    df['winning_streak'] = 0
    current_streak = 0
    for i in range(len(df)):
        if (df['close_long'].iloc[i] and df['long_returns'].iloc[i] > 0) or (df['close_short'].iloc[i] and df['short_returns'].iloc[i] > 0):
            current_streak += 1
        else:
            current_streak = 0
        df.at[i, 'winning_streak'] = current_streak
    longest_winning_streak = df['winning_streak'].max()

    # longest losing streak, defined by close_long or close_short with negative returns
    df['losing_streak'] = 0
    current_streak = 0
    for i in range(len(df)):
        if (df['close_long'].iloc[i] and df['long_returns'].iloc[i] < 0) or (df['close_short'].iloc[i] and df['short_returns'].iloc[i] < 0):
            current_streak += 1
        else:
            current_streak = 0
        df.at[i, 'losing_streak'] = current_streak
    longest_losing_streak = df['losing_streak'].max()




    info = {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "long_trades_count": long_trades_count,
        "short_trades_count": short_trades_count,
        "total_trades_count": total_trades_count,
        "long_trades_returns": long_trades_returns,
        "short_trades_returns": short_trades_returns,
        "total_trades_returns": total_trades_returns,
        "long_winning_trades": long_winning_trades,
        "long_winning_ratio": long_winning_ratio,
        "short_winning_trades": short_winning_trades,
        "short_winning_ratio": short_winning_ratio,
        "total_winning_trades": total_winning_trades,
        "total_winning_ratio": total_winning_ratio,
    }

    table_columns = ['Metric Value']
    table_data = [
        ['Sharpe Ratio', sharpe_ratio],
        ['Max Drawdown', max_drawdown],
        ['Profit Factor', profit_factor],
        ['Long Trades Count', long_trades_count],
        ['Short Trades Count', short_trades_count],
        ['Total Trades Count', total_trades_count],
        ['Long Trades Returns', long_trades_returns],
        ['Short Trades Returns', short_trades_returns],
        ['Total Trades Returns', total_trades_returns],
        ['Long Winning Trades', long_winning_trades],
        ['Long Winning Ratio', long_winning_ratio],
        ['Short Winning Trades', short_winning_trades],
        ['Short Winning Ratio', short_winning_ratio],
        ['Total Winning Trades', total_winning_trades],
        ['Total Winning Ratio', total_winning_ratio],
        ['Longest Winning Streak', longest_winning_streak],
        ['Longest Losing Streak', longest_losing_streak]
    ]
    row_labels = [table_data[i][0] for i in range(len(table_data))]
    table_data = [[table_data[i][1]] for i in range(len(table_data))]


    # output table to file
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=table_columns, rowLabels=row_labels, cellLoc='center', loc='center', colWidths=[0.5])
    table.scale(1.2, 1.2)

    cells = table.get_celld() # Get all cell objects
    for (row_idx, col_idx), cell in cells.items():
        # cell.set_height(0.15) # Set cell height for padding

        # --- Styling for Column Labels (Headers) ---
        if row_idx == 0 and col_idx >= 0:  # Targets the column header cells
            cell.set_text_props(weight='bold', color='white') # Makes text bold
            cell.set_facecolor('skyblue')

        # --- Styling for Row Labels (Leftmost Column) ---
        elif col_idx == -1 and row_idx >= 0: # Targets the row label cells
                                            # (matplotlib uses col_idx == -1 for row headers)
            cell.set_text_props(weight='bold', color='black') # Makes text bold
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
    profit_factors = [metrics['profit_factor'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(profit_factors)), profit_factors, tick_label=[f"{i + 1}" for i in range(len(profit_factors))])
    plt.title(f"Profit Factors for {name}")
    plt.xlabel('Model')
    plt.ylabel('Profit Factor')
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