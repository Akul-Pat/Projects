import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import get_data

"""Scatter-plots."""

def plot_scatter( daily_returns, SYMBOL_X, SYMBOL_Y ):
	# Scatter-plot Y vs X
	daily_returns.plot(kind='scatter', x=SYMBOL_X, y=SYMBOL_Y)

	# compute the equation of the (linear) model
	# degree of a linear polynomial is 1 - the third parameter
	linear_model = np.polyfit( daily_returns[SYMBOL_X], daily_returns[SYMBOL_Y], 1 )
	beta 	= linear_model[0]
	alpha 	= linear_model[1]

	print(f" beta of {SYMBOL_Y} = {beta:+.6f}")
	print(f"alpha of {SYMBOL_Y} = {alpha:+.6f}")

	# plot the line using the equation and SYMBOL_X as the independent variable
	plt.plot(daily_returns[SYMBOL_X], beta * daily_returns[SYMBOL_X] + alpha, '-', color='r')
	plt.legend(['Y = 0.84X + 0.000182', f'Y =  beta *X + alpha'], loc = 'upper left')
	plt.show()
	return alpha, beta


def compute_daily_returns(df):
	"""Compute and return the daily return values."""
	daily_returns = df.copy()
	daily_returns[1:] = (df[1:] / df[:-1].values) - 1
	daily_returns.iloc[0, :] = 0  # set daily returns for row 0 to 0
	return daily_returns


def test_run():
	# Read stock data from directory dataT
	# use link:
	# ln -s <SOURCE> <TARGET>
	# ln -s dataZ data
	dates = pd.date_range('2021-01-01', '2021-12-31')

	#symbols = ['SPY', 'XOM', 'GLD']
	symbols = ['SPY', 'TLT', 'GLD', 'AAPL', 'XOM', 'OXY', 'TPL',
			   'W']
	df = get_data(symbols, dates, path="dataA")


	df_new = pd.DataFrame(columns=['alpha', 'beta'], index=symbols)

	# Compute daily returns - for whole frame
	daily_returns = compute_daily_returns(df)
	print(df)
	for symbol in symbols:
		alp, bet = plot_scatter(daily_returns, 'SPY', symbol)
		df_new.loc[symbol] = [alp, bet]

	#cor_matrix = df.iloc[0, 0]

	r = daily_returns.corr()
	df_new.insert(0, 'r', r.iloc[0], allow_duplicates=False)

	df_new['alpha'] = df_new['alpha'].astype(float)
	df_new['beta'] = df_new['beta'].astype(float)
	print('test df_new')
	print(df_new)


#	Example: df = df.sort_values(by=['alpha'], ascending=True)
	#Example: print(f"\nfinal_df = {df.head(2)}\n\n{df.tail(2)}")


	print("======================================")
	df_new = df_new.sort_values(by=['alpha'], ascending=True)
	df_new.fillna(0)
	print(f"\nfinal_df = {df_new.head(2)}\n\n{df_new.tail(2)}")




	# first row are stock correlated with SPY -
	# print the correlation matrix.
	print( daily_returns.corr(method='pearson') )

	
if __name__ == "__main__":
	test_run()

"""==============================================================================="""	
