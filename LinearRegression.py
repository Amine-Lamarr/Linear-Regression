from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.metrics import mean_absolute_error , mean_squared_error 
import pandas as pd
import matplotlib.pyplot as plt

# preparing data  
data = pd.read_csv(r"C:\Users\lenovo\Downloads\salarieszip\Salary_dataset.csv")
data = data.drop('Unnamed: 0' , axis=1)
null = data.isnull().sum()
print("any null values :\n" , null)
x = data['YearsExperience'].to_numpy().reshape(-1 , 1)
y = data['Salary'].to_numpy()

# showing data 
plt.style.use("fivethirtyeight")
plt.scatter(x , y , c=y , cmap='cool' , s=300 ,  label = 'salaries')
plt.colorbar()
plt.legend()
plt.xlabel('years')
plt.ylabel('salaries')
plt.title("years experince & salary")
plt.show()

# scalling "Salary" features
scalling = StandardScaler()
x_scalled = scalling.fit_transform(x)

# splitting data 
x_train , x_test , y_train , y_test = train_test_split(x_scalled , y , test_size=0.2 , random_state=32)

# model 
model = LinearRegression()
model.fit(x_train , y_train)

# model 2 
model2 = Lasso(alpha=0.7)
model2.fit(x_train , y_train)

# model 3
model3 = Ridge(alpha=0.1)
model3.fit(x_train , y_train)

# predictions 
scaled_3 = scalling.transform([[3]])
predictions = model.predict(x_test)
predict = model.predict(scaled_3)

predictions2 = model2.predict(x_test)
predict2 = model2.predict(scaled_3)

predictions3 = model3.predict(x_test)
predict3 = model3.predict(scaled_3)

# results 
train_score = model.score(x_train , y_train)
test_score = model.score(x_test , y_test)
mse = mean_squared_error(y_test , predictions )
mae = mean_absolute_error(y_test , predictions )
coef = model.coef_
interc = model.intercept_

# results 2 
train_score2 = model2.score(x_train , y_train)
test_score2 = model2.score(x_test , y_test)
mse2 = mean_squared_error(y_test , predictions2 )
mae2 = mean_absolute_error(y_test , predictions2 )
coef2 = model2.coef_
interc2 = model2.intercept_

# results 3 
train_score3 = model3.score(x_train , y_train)
test_score3 = model3.score(x_test , y_test)
mse3 = mean_squared_error(y_test , predictions3 )
mae3 = mean_absolute_error(y_test , predictions3 )
coef3 = model3.coef_
interc3 = model3.intercept_

# printing results 
print("linear regresison model : \n ")
print(f"score (train) : {train_score*100:.2f}%")
print(f"score (test) : {test_score*100:.2f}%") 
print(f"mean absolute error : " , mae)
print(f"mean squared erorr : " , mse)
print(f"model predicts 3y exp = {predict}$")
print(coef)
print(interc)
print()
print("lasso regression : \n")

# results 2
print(f"score (train) : {train_score2*100:.2f}%")
print(f"score (test) : {test_score2*100:.2f}%") 
print(f"mean absolute error : " , mae2)
print(f"mean squared erorr : " , mse2)
print(f"model predicts 3y exp = {predict2}$")
print(coef2)
print(interc2)
print()
print("ridge regression : \n")

# results 3
print(f"score (train) : {train_score3*100:.2f}%")
print(f"score (test) : {test_score3*100:.2f}%") 
print(f"mean absolute error : " , mae3)
print(f"mean squared erorr : " , mse3)
print(f"model predicts 3y exp = {predict3}$")
print(coef3)
print(interc3)