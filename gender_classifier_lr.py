
# X = [height,weight,shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37],[166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], 
     [181, 85, 43], [168, 75, 41], [168, 77, 41]]

Y = ["male", "male", "female",
    "female", "male", "male","female",
    "female","female", "male",
    "male","female", "female"]

test_data = [[190, 70, 43],[154, 50, 37],[160,65,44],[150,90,44],[154, 50, 35]]
test_labels = ["male","female","male","male","female"]

#LogisticRegression
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()
lr_clf = lr_clf.fit(X,Y)
lr_prediction = lr_clf.predict(test_data)

print('Logistic Regression Result = ',lr_prediction)

import numpy as np
from sklearn.metrics import accuracy_score

acc = accuracy_score(lr_prediction,test_labels)

print('accuracy score = ',acc)
print(lr_clf.score(test_data,test_labels))