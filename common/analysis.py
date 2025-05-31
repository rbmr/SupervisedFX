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
    equity_open = df['info.agent_data.equity.open'].values
    equity_high = df['info.agent_data.equity.high'].values
    equity_low = df['info.agent_data.equity.low'].values
    equity_close = df['info.agent_data.equity.close'].values
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
    returns = df['info.agent_data.equity.close'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std()

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