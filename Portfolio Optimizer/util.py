""" Utility code."""

import os
import pandas as pd
import matplotlib.pyplot as plt

def symbol_to_path( symbol, path="data" ):
    dir = path
    base_dir = os.path.join("", dir)
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid(color='tab:grey', linestyle='-', linewidth=0.2)
    plt.show()


def get_data(symbols, dates, addSPY=True, colname = 'Adj Close', path="data"):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol, path), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        # print(df_temp.head(2))
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
        # print(df.head(2))
    return df



