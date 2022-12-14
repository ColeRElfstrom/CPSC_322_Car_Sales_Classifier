import numpy as np

from mysklearn import myutils
from mysklearn import myevaluation
from myclassifiers import MyRandomForestClassifier

def test_MyRandomForestClassifier_fit():
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]

    X = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    table = [X[i] + [y[i]] for i in range(len(X))]


    X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, shuffle=True)
    remainder_set = []
    for i, row  in enumerate(X_train):
        row.append(y_train[i])
        remainder_set.append(row)

    rf_clf = MyRandomForestClassifier()
    rf_clf.fit(remainder_set, 100, 20, 4)

    for tree in rf_clf.tree_list:
        print(tree.tree)

    
    assert len(rf_clf.tree_list) == 20



def test_MyRandomForestClassifier_predict():


    # TODO
    assert False is True