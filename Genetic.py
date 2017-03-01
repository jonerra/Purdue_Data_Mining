# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from gplearn.genetic import SymbolicRegressor

# Importing the input and target data sets
df = pd.read_csv('credit_data.csv')
target = pd.read_csv('DataminingContest2009.Task2Targets.Train.csv')

# Subset the input variables to only the most important
df = df[['field3', 'flag5', 'field4', 'zipbins', 'amount']]

# Inputting the state missing value with the most common state
df['zipbins'] = df['zipbins'].fillna(df['zipbins'].mode()[0])

# Convert our data frame to multidimensional arrays
# The X variable is an array of the independent variables and drops column 'y'
X = np.array(df)
X = preprocessing.scale(X)

# The y variable is an array of the dependent variable 'y'
y = np.array(target['fraud'])

# Shuffle and partition our data into 80% train data and 20% test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

est_gp = SymbolicRegressor(population_size=4000,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

est_gp.fit(X_train, y_train)

accuracy = est_gp.score(X_test, y_test)
print(accuracy)