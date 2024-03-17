import argparse
from dataset import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('--classifier', default='GaussianNB', 
                     choices=['GaussianNB', 
                              'SVC', 
                              'KNeighborsClassifier',
                              'GradientBoostingClassifier',
                              'DecisionTreeClassifier',
                              'XGBClassifier',
                              'AdaBoostClassifier',
                              'RandomForestClassifier'])
    arg.add_argument('--X', default=X)
    arg.add_argument('--y', default=y)
    return arg.parse_args()

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    # Train the classifier
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred_test = classifier.predict(X_test)
    
    # Confusion matrix for test set
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    # Predicting the Training set results
    y_pred_train = classifier.predict(X_train)
    
    # Confusion matrix for training set
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    # Calculate accuracy for training set
    accuracy_train = np.round((cm_train[0][0] + cm_train[1][1]) / len(y_train), 2)
    
    # Calculate accuracy for test set
    accuracy_test = np.round((cm_test[0][0] + cm_test[1][1]) / len(y_test), 2)
    
    return accuracy_train, accuracy_test

if __name__ == "__main__":
    args = arguments()

    X_train, X_test, y_train, y_test = train_test_split(args.X, args.y, test_size=0.2, random_state=42)
    
    classifiers = {
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'),
        'SVC': SVC(kernel='rbf', random_state=42),
        'GaussianNB': GaussianNB(),
        'XGBClassifier': XGBClassifier(objective="binary:logistic", random_state=42, n_estimators = 100),
        'GradientBoostingClassifier': GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, max_depth=3, random_state=42),
        'DecisionTreeClassifier':DecisionTreeClassifier(criterion='gini'),
        'AdaBoostClassifier':AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
        'RandomForestClassifier':RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=2, n_estimators = 10, random_state=42)

    }

    selected_classifier = classifiers[args.classifier]
    print("Evaluating {}:".format(args.classifier))
    accuracy_train, accuracy_test = evaluate_classifier(selected_classifier, X_train, y_train, X_test, y_test)
    print('Accuracy for training set: {}'.format(accuracy_train))
    print('Accuracy for test set: {}'.format(accuracy_test))
