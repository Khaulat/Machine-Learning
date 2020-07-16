# Import libraries/dependencies

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
%mtplotlib inline
import sklearn
import statsmodels.api as sm


# Load dataset

from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()


# Understand data

boston.data.shape
print boston.feature_names
print boston.DESCR
data = pd.DataFrame(boston.data)
data.head()

data.columns = boston.feature_names
data.head()


# Define target

data['PRICE'] = boston.target
data.head()


# Visualizing dataset and the relatonship of some input variables to the output(price)

data.describe()

plt.scatter(data.CRIM, data.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")
plt.savefig('./CRIMxPRICE.png', dpi=400)

plt.scatter(data.RM, data.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
plt.savefig('./RMxPRICE.png', dpi=400)

plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")
plt.savefig('./PTRATIOxPRICE.png', dpi=400)

# We drop the price from the original dataset as it is the target
X = data.drop('PRICE', axis = 1)

# Train model 

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

model = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(model, parameters, scoring = 'neg_mean_squared_error', cv = 5)

lasso_regressor.fit(X, data.PRICE)


# Checking for the error( in this case, Mean Squared Error)

MSE = mean_squared_error(data.PRICE, model.predict(X))
print MSE