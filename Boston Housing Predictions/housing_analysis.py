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

#filter out warning message.
warnings.filterwarnings("ignore", category=FutureWarning)


#import the data set
#the dataset used is a kaggle dataset that gives various features about
#the boston real estate market and its surrounding area.
data = 'Housing.csv'

#read in csv
df = pd.read_csv(data)

#start to clean data
#first to delete empty and duplicate values

df = df.dropna()
df = df.drop_duplicates()

#converting float to int
df['CHAS'] = df['CHAS'].astype(float)
df["AGE"] = df["AGE"].astype(int)
df["ZN"] = df["ZN"].astype(int)
df["RAD"] = df["RAD"].astype(int)
df["TAX"] = df["TAX"].astype(int)

#start to visualize and manipulate the data
#first without any external computations, we can plot the data and visually see what is going on in
#the housing spreadsheet.

#generate a heat map to see relations from the correlation matrix

print(df.head())

#corr heat map
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)
plt.title("Correlations Between Variables",size=15)
plt.show()


#plot histograms for every column catagory.
#also add them all to one figure to make it easier to view
fig = plt.figure(figsize=(16,16))
for i, col in enumerate(df.columns):
    ax = fig.add_subplot(len(df.columns)//2, 2, i+1)
    ax.hist(df[col], color='Red')
    ax.set_title(col)
    ax.set_ylabel('Prevalence')
    plt.tight_layout()

plt.show()

#more data
#plots showing the correlation they have with each of the following columns:
#rooms, median value, zone, employment distance, status of lower income people
from pandas.plotting import scatter_matrix
a = ["RM","MEDV","ZN","DIS","LSTAT"]
scatter_matrix(df[a],color= "b",alpha=0.8,figsize=(15,10))
plt.show()



#use sklearn and its some of its pre-trained models to do an analysis on the dataset
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# prepare the data for training the model
X = train_set.drop(["LSTAT","MEDV"], axis=1)
Y = train_set["MEDV"].copy()

#get training and test variables
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#normalize code. features with highest impact on biases
a = ['ZN', 'INDUS', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B']
scale = StandardScaler()
x_train[a] = scale.fit_transform(x_train[a])
x_test[a] = scale.fit_transform(x_test[a])
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

#calculate the cross validation score
def rmse_cv(model):
    score = cross_val_score(model,x_train,y_train,cv=5,scoring="neg_mean_squared_error")
    score_cv = np.sqrt(-score)
    score_cv_mean = score_cv.mean()
    return score_cv_mean

# Linear Regression model
linreg = LinearRegression()
linreg.fit(x_train, y_train)
y_pred = linreg.predict(x_test)
linreg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
linreg_mae = mean_absolute_error(y_test, y_pred)

#  Decision Tree Regression model
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
y_pred = dtr.predict(x_test)
dtr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
dtr_mae = mean_absolute_error(y_test, y_pred)

#  Random Forest Regression model
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)
rfr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rfr_mae = mean_absolute_error(y_test, y_pred)

# Bagging Regression model
bag = BaggingRegressor()
bag.fit(x_train, y_train)
y_pred = bag.predict(x_test)
bag_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
bag_mae = mean_absolute_error(y_test, y_pred)

# Print results
print("Linear Regression RMSE:", linreg_rmse)
print("Linear Regression MAE:", linreg_mae)
print("Decision Tree Regression RMSE:", dtr_rmse)
print("Decision Tree Regression MAE:", dtr_mae)
print("Random Forest Regression RMSE:", rfr_rmse)
print("Random Forest Regression MAE:", rfr_mae)
print("Bagging Regression RMSE:", bag_rmse)
print("Bagging Regression MAE:", bag_mae)



#restructure code to plot the results of the output


#list of models to be used.
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree Regressor', DecisionTreeRegressor(random_state=42)),
    ('Random Forest Regressor', RandomForestRegressor(random_state=42)),
    ('Bagging Regressor', BaggingRegressor(base_estimator=DecisionTreeRegressor(random_state=42), random_state=42))
]

#dummy var to hold the outputs
results = []
score = 0
#iterate through the models.
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    #recalcuate mse
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    score = rmse_cv(model)
    #append
    results.append({'name': name, 'rmse': rmse, 'score' : score, 'mae': mae,'y_pred': y_pred, 'y_test': y_test})
    print(f"Cross Valuation Score of model {name} : ",score)

# plot the results
plt.figure(figsize=(12, 8))

# plot bar chart of RMSE and MAE
plt.subplot(2, 2, 1)
sns.barplot(x='name', y='rmse', data=pd.DataFrame(results))
plt.title('RMSE of different models')
plt.ylabel('RMSE')
plt.ylim(0, 10)
plt.xticks(fontsize=5)

plt.subplot(2, 2, 2)
sns.barplot(x='name', y='mae', data=pd.DataFrame(results))
plt.title('MAE of different models')
plt.ylabel('MAE')
plt.ylim(0, 10)
plt.xticks(fontsize=5)

plt.subplot(2, 2, 3)
sns.barplot(x='name', y='score', data=pd.DataFrame(results))
plt.title('Cross_Validation of different models')
plt.ylabel('Score')
plt.ylim(0, 10)

#change font size
plt.xticks(fontsize=5)
plt.tight_layout()
plt.show()

from sklearn import datasets
b_data = datasets.load_boston()
target = b_data.target

#plot the preditction against the best performing model
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
y_pred = cross_val_predict(rfr, b_data.data, target, cv=10)

#plot chart
fig, ax = plt.subplots()
ax.scatter(target, y_pred, edgecolors=(0, 0, 0))
ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
ax.set_title('Prediction Chart Random Forest Regressor')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

print("done")
exit(0)



