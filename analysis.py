import pandas as pd 
data = pd.read_csv('student_data.csv')

print(data.head())
print(data.info())
print("Columns Are:")
print(data.columns)
#x= data.drop("G3", axis= 1)
x=data[["studytime","failures", "G1","G2"]].astype(int)
y=data["G3"]

print("shape of Xx=", x.shape)
print("shape of y=", y.shape)

#x= pd.get_dummies(x,drop_first=True)
#print(x.head()) #One-Hot Encoding. 

#Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print("shape of x_train=", x_train.shape)
print("shape of x_test=", x_test.shape)
print("shape of y_train=", y_train.shape)
print("shape of y_test=", y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#Model Prediction

y_pred= model.predict(x_test)
print("Predicted Values are:", y_pred)

#Model Accuracy and performance test 
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score
import numpy as np

mae= mean_absolute_error(y_test,y_pred)
mse= mean_squared_error(y_test,y_pred)
rsme= np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rsme)
print("R2 Score:", r2)

#actual vs predicted.
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()

#Ridge Regression.
from sklearn.linear_model import Ridge
model_ridge= Ridge(alpha=1.0)
model_ridge.fit(x_train,y_train)
y_pred_ridge= model_ridge.predict(x_test)
print("Ridge R2:", r2_score(y_test, y_pred_ridge))


#Lasso Regression.
from sklearn.linear_model import Lasso
model_lasso= Lasso(alpha=0.1)
model_lasso.fit(x_train,y_train)
y_pred_lasso= model_lasso.predict(x_test)
print("Lasso R2:", r2_score(y_test, y_pred_lasso))

#model Save
import pickle
pickle.dump(model_lasso, open("model.pkl", "wb"))
