from sklearn import tree
#0-motorcycle 1-car
labels=[0,0,1,0,1,1,0,0]
features=[[2,2],[2,2],[3,4],[2,2],[4,4],[4,4],[3,2],[2,2]]

algorithm=tree.DecisionTreeClassifier()
algorithm=algorithm.fit(features,labels)
inputdata=[[4,4]]

print(algorithm.predict(inputdata))