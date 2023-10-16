"""Bollinger Bands."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    return df


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    # TODO: Compute and return the rolling mean

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

    return_data = values
    return_data = return_data.rolling(window= window).mean()
    return return_data


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # TODO: Compute and return rolling standard deviation
    # return pd.rolling_std(values, window=window)

    return_data = values
    return_data = return_data.rolling(window= window).std()
    return return_data


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # TODO: Compute upper_band and lower_band using rm - and rstd
    # 2 standard deviations aboav the mean.
    # add 2 times the rolling standard deviations of the mean 2*std.'

    upper_band = rm - 2*rstd
    lower_band = 2*rstd + rm

    return upper_band, lower_band


def test_run():
    # Read data
    dates = pd.date_range('2020-01-01', '2021-05-31')
    symbol_string = 'TSLA'
    symbols = [symbol_string]

    df = get_data(symbols, dates)

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm = get_rolling_mean( df[symbol_string], window=10 )

    # 2. Compute rolling standard deviation
    rstd = get_rolling_std( df[symbol_string], window=10 )

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands( rm, rstd )

    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = df[symbol_string].plot(title="Bollinger Bands", label=symbol_string, color='blue', xlabel="Date")
    rm.plot(label='Rolling mean', color='red', linestyle = 'dotted', linewidth = 0.5)
    upper_band.plot(label='Upper band')
    lower_band.plot(label='Lower band')

    # 4. Make plot look pretty
    # TODO:  USE plt.fill_between the upper band and lower band
    # plt.fill_between  the upper band and lower band
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html

    #making the signal frame, data frame
    sf = pd.DataFrame()
    sf['Upper'] = upper_band
    sf['Lower'] = lower_band
    sf[symbol_string] = df[symbol_string].values
    sf.dropna(how='any', inplace=True)



    #making the buy signal frame, data frame
    bsf = sf.iloc[list((sf[symbol_string].values) < (sf['Lower'].values))]
    bsf = bsf.loc[:, lambda bsf:[symbol_string]]
    bsf.columns = ['Price']

    #create the sell signalframe, datat frame
    ssf = sf.iloc[list((sf[symbol_string].values) > (sf['Upper'].values))]
    ssf = ssf.loc[:, lambda ssf: [symbol_string]]
    ssf.columns = ['Price']


    #buy = sf.loc[list(sf.Upper)]
    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.grid( color='grey',linestyle = '--',linewidth = 0.3 )
    plt.fill_between(df.index ,lower_band, upper_band, color = 'grey',alpha = 0.5)
    plt.scatter(x= ssf.index, y = ssf.Price, alpha = 0.95, color = 'green')
    plt.scatter(x= bsf.index , y = bsf.Price ,alpha=0.6,color = 'red' )
    plt.show()

if __name__ == "__main__":
    test_run()
