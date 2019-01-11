#binary tree
from sklearn import tree

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38],
     [154, 54, 37], [166, 65, 40], [190, 90, 47],
     [175, 64, 39], [177, 70, 40], [159, 55, 37],
     [171, 75, 42], [181, 85, 43]]

# genders
Y = ['male', 'female', 'female',
     'female', 'male', 'male',
     'male', 'female', 'male',
     'female', 'male']

# decision tree classifier
clf = tree.DecisionTreeClassifier()

# train against data set
clf = clf.fit(X, Y)

# make a prediction for a new set of values
prediction = clf.predict([[190, 100, 42]])

# show predictions
print(prediction)