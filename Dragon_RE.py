#Author : Deepansh Dubey.
#Date   : 10/10/2021.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load

housing = pd.read_csv("data.csv")

housing.info()
housing['CHAS'].value_counts()
housing.describe()

#Plotting Histograms

housing.hist(bins=50, figsize=(20,15))
plt.show()

# Train-Test Splitting
#Manual data splitting

def split_train_test(data, test_ratio):
    shuffled = np.random.permutation(len(data))
    np.random.seed(42)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")

#SPlitting data using sklearn

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")

#Stratified shuffle split

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set['CHAS'].value_counts()
strat_train_set['CHAS'].value_counts()
housing = strat_train_set.copy()

# Looking for Correlations

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


attributes = ["RM", "ZN", "MEDV", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))

housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

# Trying attribute combinations

housing["TAXRM"] = housing["TAX"]/housing["RM"]
housing["TAXRM"]
housing.head()

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)
housing.describe()
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Feature scaling
# Pipeline

my_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('std_scaler', StandardScaler())])
housing_num_tr = my_pipeline.fit_transform(housing)

# Selecting a Model
housing_num_tr.shape
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)
print(list(some_labels))

# Evaluating the Model
housing_predictions = model.predict(housing_num_tr)
d_mse = mean_squared_error(housing_labels, housing_predictions)
d_rmse = np.sqrt(d_mse)
print(d_rmse)

# Cross-Validation
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
print_scores(rmse_scores)

# Saving The Model
dump(model, "Dragon.joblib")

# Model Testing
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

# Using The Model
model = joblib.load('Dragon.joblib', mmap_mode=None)
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)
