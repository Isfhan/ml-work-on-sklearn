
# X = [height,weight,shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37],[166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], 
     [181, 85, 43], [168, 75, 41], [168, 77, 41]]

Y = ["male", "male", "female",
    "female", "male", "male","female",
    "female","female", "male",
    "male","female", "female"]

test_data = [[190, 70, 43],[154, 50, 37],[160,65,44],[150,90,44]]
test_labels = ["male","female","male","male"]

#DecisionTreeClassifier
from sklearn import tree

dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(X,Y)
dtc_prediction = dtc_clf.predict(test_data)

print('Decision Tree Classifier Result = ',dtc_prediction)

import numpy as np
from sklearn.metrics import accuracy_score

acc = accuracy_score(dtc_prediction,test_labels)

print('accuracy score = ',acc)
print(dtc_clf.score(test_data,test_labels))