import pandas as pd
import numpy as np
import datetime as dt

# these are provided in file: util.py
from util import get_data, plot_data

def plot_normalized_data(df,
                         title="Normalized prices",
                         xlabel="Date",
                         ylabel="Normalized price"
                         ):
    """
    Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fn = "[plot_normalized_data]: "
    df_normed = df / df.iloc[0, :]
    plot_data(df_normed, title=title, xlabel=xlabel, ylabel=ylabel)


def get_portfolio_stats(port_val,
                        daily_rf=0.0,
                        samples_per_year=252.0):


    fn = "[get_portfolio_stats]: "

    current = port_val
    prev = port_val.shift(1)
    daily_rets = ((current - prev)/ prev)
  #  daily_rets = daily_rets[1:]

    daily_rets      = daily_rets.iloc[1:]

    avg_daily_ret   = daily_rets.mean()
    std_daily_ret   = daily_rets.std()

    # replace 0.0 by eqation for cum_ret.
    start = daily_rets.iloc[0]
    final = daily_rets.iloc[-1]
    cum_ret         = prev.iloc[-1]/prev.iloc[0] - 1

    sharpe_ratio    = 0.0
    sharpe_ratio = np.sqrt(samples_per_year) * np.mean(daily_rets - daily_rf) / std_daily_ret

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def get_portfolio_value(prices,
                        allocs=None,
                        start_val=1000000
                        ):

    if allocs is None:
        allocs = [0.1, 0.2, 0.3, 0.4]

    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * start_val

    port_val = alloced.sum(axis=1)
    ## Make more stuff print out
    pd.options.display.float_format = '{:20,.2f}'.format
    #print(port_val)

    return port_val


def assess_portfolio(
        plot    = True,
        sd      = dt.datetime(2008, 1, 1),  # start time
        ed      = dt.datetime(2009, 1, 1),  # end time
        syms    = None,                     # symbols = []
        allocs  = None,                     # allocation []
        sv      = 1000000,                  # starting value/Capital
        rfr     = 0.0,                      # riskfree rate
        sf      = 252.0,                    # samling frequency
):
    fn = "[asses_portfolio]: "

    # mutable - arguments = replace by defaults if lacking
    if allocs is None:
        allocs = [0.1, 0.2, 0.3, 0.4]
    if syms is None:
        syms = ['GOOG', 'AAPL', 'GLD', 'XOM']

    print("*** ASSESS PORTFOLIO *** ")

    cr      = 0  # cumulative return
    adr     = 0  # average daily return
    sddr    = 0  # volatility (standard deviation)
    sr      = 0  # sharpe ratio  ---> will be given
    ev      = 0  # end value of portfolio

    dates           = pd.date_range( sd, ed )

    prices_all      = get_data( syms, dates, path="data0" )    # automatically adds SPY

    print(f"\nPrices of symbols in portfolio (+ SPY if not in portfolio): top& bottom 2 lines: ")
    print(f"{prices_all.head(2)}")
    print("...")
    print(f"{prices_all.tail(2).to_string(header=False)}")

    prices      = prices_all[syms]   # cleaning up so we get our  portfolio symbols
    prices_SPY  = prices_all['SPY']  # only SPY, for comparison later - Series


    port_val = prices_SPY

    port_val = get_portfolio_value(prices, allocs, sv)  # better to use a method for this.


    ev = port_val[-1]  # end value is the last value.
    cr, adr, sddr, sr = get_portfolio_stats(port_val, rfr, sf)


    if plot:
        # add code to plot here - GIVEN
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp, title="Daily portfolio value and SPY")
        pass

    return cr, adr, sddr, sr, ev


class InitPorfolio:
    def __init__(self,
                 plot           = True,
                 start_date     = dt.datetime(2010, 1, 1),
                 end_date       = dt.datetime(2010, 12,31),
                 symbols        = None,
                 allocations    = None,
                 start_value    = 5000,
                 risk_free_rate = 0.0,
                 m_trading_days = 252
                 ):

        if symbols is None:
            symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

        if allocations is None:
            # defaul to allocations spread evenly across symbols
            length = len(symbols)
            allocations = [1.0 / length] * length
            allocations = [0.25, 0.25, 0.25, 0.25]

        self.plot           = plot
        self.start_date     = start_date
        self.end_date       = end_date
        self.symbols        = symbols
        self.allocations    = allocations
        self.start_value    = start_value
        self.risk_free_rate = risk_free_rate
        self.m_trading_days = m_trading_days

        # correcting values in default/ initialization - could bonk out.
        if len(self.allocations) !=  len(self.symbols):
            length = len(self.symbols)
            allocations = [1.0 / length] * length
            print("resetting allocation ... \n")
            self.allocations    = allocations

    # override default to output each element in object instance
    def __str__(self):
        # s0 = f"__str__"
        s0 = f""
        s2 = f"Plot:       \t\t{self.plot}"
        s3 = f"Start Date: \t\t{self.start_date:%Y-%m-%d}"
        s4 = f"End Date:   \t\t{self.end_date:%Y-%m-%d}"
        s5 = f"Symbols:    \t\t{self.symbols}"
        s6 = f"Allocations:\t\t{self.allocations}"
        s7 = f"Start Value:\t\t{self.start_value:,d}"
        s8 = f"Risk Free Rate:\t\t{self.risk_free_rate}"
        s9 = f"Trading Days:\t\t{self.m_trading_days}"
        str_string_of_class    = f"{s2}\n{s3}\n{s4}\n{s5}\n{s6}\n{s7}\n{s8}\n{s9}"

        return str_string_of_class


def test_code(args):
    # args is from init
    fn=f"test_code"


    # TEST assess_portfolio with  the values from args sent in as a parameter
    cr, adr, sddr, sr, ev = assess_portfolio(
        plot    = args.plot,
        sd      = args.start_date,
        ed      = args.end_date,
        syms    = args.symbols,
        allocs  = args.allocations,
        sv      = args.start_value
        )

    # Print statistics -----------------------
    print("")
    print(f"---- Input ----{'-'*45}")
    print(args)

    print(f"---- Result ---{'-'*45}")
    print(f"Sharpe Ratio:\t\t{sr:8.6f}")
    print(f"Volatility (stdev)\t{sddr:8.6f}")
    print(f"Avg Daily Return\t{adr:8.6f}")
    print(f"Cum Return: \t\t{cr:8.6f}")
    print(f"{'-'*60}")



if __name__ == "__main__":
    # other check args = 0

    # create AN instances - see defaults in InitPortfolio- set up for now.
    arg_processing = InitPorfolio( # set to defaults
        plot                = True,
        start_date          = dt.datetime(2021, 1, 1),
        end_date            = dt.datetime(2021, 12, 31),
        # symbols = symbols,  - use default - see if it works ...
        symbols             = ['TPL', 'OXY'],
        # symbols           =  ['AXP', 'HPQ', 'IBM', 'HNZ'],
        allocations         = [.5, .5],
        # allocations         = [0.0, 0.0, 0.0, 1],
        start_value         = 5000,
        risk_free_rate      = 0,
        m_trading_days      = 252
    )
    # print(arg_processing)

    test_code( arg_processing )

