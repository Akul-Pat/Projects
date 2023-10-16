Required Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

how to run code:
all the files needed are already in the directory. The only change that needs to be made is to change/set the
path for the Housing.csv in housing_analysis.py. Once that is done, run the code and you will get an output for
the RMSE, MAE, and CVS, and all the plots and charts that were generated.