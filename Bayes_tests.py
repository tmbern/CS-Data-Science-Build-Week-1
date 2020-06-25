import numpy as np
from scipy import stats
from BayesClassifier import BayesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets

# get the X and y datasets to test
X, y = datasets.load_breast_cancer(return_X_y=True)

print(f'Features Shape: {X.shape}')
print(f'Target Shape: {y.shape}')

# split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

print('Bayes Class implementation')
# instantiate the Bayes Classifier
model = BayesClassifier()
# fit the model
model.fit(X_train, y_train)
# print("The model has been trained")
# get predictions
predictions = model.predict(X_test)
# print("predictions have been made")
# print(predictions)
# print(y_test)
# # get the accuracy
# print(len(predictions) == len(y_test))
num_el, counts_el = np.unique( y_test == predictions, return_counts=True)
# print(num_el)
# print(counts_el)
accuracy = counts_el[1] / sum(counts_el)
print(f' the accuracy is: {accuracy}')


# Get accuracy for sklearn model
print('Sklearn GaussianNB model')

model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
num_el, counts_el = np.unique( y_test == predictions, return_counts=True)
accuracy = counts_el[1] / sum(counts_el)
print(f' the accuracy is: {accuracy}')