#import the package and load the algorithm
from sklearn import tree

#creating Training Data(features,labels)


#lables are class outcomes(0,1)
labels=[0,0,1,1,0]

#features are measurements(leg and weight)
features=[[0,50],[0,150000],[4,20],[4,23],[0,0.45]]

#initialize the classification Algorithm named DecisionTree

algorithm=tree.DecisionTreeClassifier()

#train the algorithm with training data(features and labels)
algorithm=algorithm.fit(features,labels)

#predict - new data [4 legs, 32kg]
inputdata=[[4,32]]
#predict fish(0) or cat(1)
print(algorithm.predict(inputdata))