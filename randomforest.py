from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import data_blitz
from matplotlib import pyplot as plt

X_train = data_blitz.X_train
Y_train = data_blitz.Y_train
X_test = data_blitz.X_test
Y_test = data_blitz.Y_test

model = RandomForestRegressor(n_estimators=10)
model.fit(X_train, Y_train)

test_predict = model.predict(X_test)

# print(test_predict)

# getting mean squared error
error = mean_squared_error(Y_test, test_predict, squared=False)
print(error)

# plot it in graph for reference

Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label='Actual')
plt.plot(test_predict, color='red', label='Predicted')
plt.show()

# as mean squared error is leeser with random forest model was suited more than linear model