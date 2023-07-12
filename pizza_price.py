from sklearn.linear_model import LinearRegression

x=[[4],[8],[12],[16],[18]]
y=[[4],[8],[10],[12],[15]]

model=LinearRegression()
model.fit(x,y)

diameter=13

predict=model.predict([[diameter]])
print("The price of 13inch pizza is %.2f"%predict)