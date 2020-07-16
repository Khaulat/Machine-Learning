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


# Train model - We drop the price from the original dataset as it is the target.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = data.drop('PRICE', axis = 1)

model = LinearRegression()
model.fit(X, data.PRICE)


# Now predicting and plotting the prices

model.predict(X)[0:5]

plt.hist(data.predict(X))
plt.title('Predicted Housing Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')


# Checking for the error( in this case, Mean Squared Error)

MSE = mean_squared_error(data.PRICE, model.predict(X))
print MSE