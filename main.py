import pandas as pd
import numpy as np

data = pd.read_csv("iris.csv")

data["species"] = data["species"].map({"setosa":0, "versicolor":1,"virginica":2})
x = data[["sepal_length","sepal_width","petal_length","petal_width"]]
y = data[["species"]]

print(data.head())
print(data.info())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 5)

print(x_train.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

from sklearn.metrics import mean_squared_error

y_predict = lr.predict(x_test)

rmse_lr = (np.sqrt(mean_squared_error(y_test,y_predict)))
print("the rmse of iris dataset is: ",rmse_lr)

