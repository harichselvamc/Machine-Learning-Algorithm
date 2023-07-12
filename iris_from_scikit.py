from sklearn import tree
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris=load_iris()


# # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# # print(iris.feature_names)
# # print(iris.target_names)
# # print(iris.data[0])
# # print(iris.target[0])


x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
algo=tree.DecisionTreeClassifier()
algo.fit(x_train,y_train)

# predictions=algo.predict(x_test)
# print(accuracy_score(y_test,predictions))
newdata=[[5.0,3.5,1.4,0.2]]
predictions=algo.predict(newdata)
print(predictions)


