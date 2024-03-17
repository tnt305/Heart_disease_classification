from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
import argparse
from dataset.dataset import *

def arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('--classifier', default='SVC', choices=['SVC', 'KNeighborsClassifier', 'GradientBoostingClassifier', 'DecisionTreeClassifier', 'XGBClassifier', 'AdaBoostClassifier', 'RandomForestClassifier'])
    arg.add_argument('--X', default=X)
    arg.add_argument('--y', default=y)
    return arg.parse_args()

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred_test = classifier.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_train, y_pred_train)
    accuracy_train = np.round((cm_train[0][0] + cm_train[1][1]) / len(y_train), 2)
    accuracy_test = np.round((cm_test[0][0] + cm_test[1][1]) / len(y_test), 2)
    return accuracy_train, accuracy_test

if __name__ == "__main__":
    args = arguments()
    X_train, X_test, y_train, y_test = train_test_split(args.X, args.y, test_size=0.2, random_state=42)

    classifiers = {
        'SVC': SVC(kernel='rbf', random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'XGBClassifier': XGBClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42)
    }

    if args.classifier == 'StackingClassifier':
        xg = XGBClassifier()
        final_classifier = StackingClassifier(estimators=[('dtc', DecisionTreeClassifier(random_state=42)),
                                                          ('rfc', RandomForestClassifier(random_state=42)),
                                                          ('knn', KNeighborsClassifier()),
                                                          ('gc', GradientBoostingClassifier(random_state=42)),
                                                          ('ad', AdaBoostClassifier(random_state=42)),
                                                          ('svc', SVC(kernel='rbf', random_state=42))],
                                              final_estimator=xg)
    else:
        final_classifier = classifiers[args.classifier]

    print("Evaluating {}:".format(args.classifier))
    accuracy_train, accuracy_test = evaluate_classifier(final_classifier, X_train, y_train, X_test, y_test)
    print('Accuracy for training set: {}'.format(accuracy_train))
    print('Accuracy for test set: {}'.format(accuracy_test))
