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

    # calculate sharpe ratio on the close prices of equity
    returns = df['info.agent_data.equity_close'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std()

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

    return {
        "sharpe_ratio": sharpe_ratio,
    }

def analyse_finals(final_metrics: List[Dict[str, Any]], results_path: Path, name: str) -> None:
    """
    Analyze the final results DataFrame and save the analysis to the results_path.
    """
    # Ensure results_path exists
    results_path.mkdir(parents=True, exist_ok=True)

    # make a plot of the sharpe ratios
    sharpe_ratios = [metrics['sharpe_ratio'] for metrics in final_metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sharpe_ratios)), sharpe_ratios, tick_label=[f"Model {i + 1}" for i in range(len(sharpe_ratios))])
    plt.title(f"Sharpe Ratios for {name}")
    plt.xlabel('Model')
    plt.ylabel('Sharpe Ratio')
    plt.savefig(results_path / f"sharpe_ratios.png")
    plt.close()