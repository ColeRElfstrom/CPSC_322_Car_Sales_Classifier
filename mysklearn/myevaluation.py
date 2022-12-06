import copy
import random
from mysklearn import myutils
import numpy as np

# TODO: copy your myevaluation.py solution from PA5 here
def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if isinstance(test_size, float):
        index_for_split = int(len(y)*(1-test_size))
    elif isinstance(test_size, int):
        index_for_split = len(y) - test_size
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    y_copy = copy.deepcopy(y)
    X_copy = copy.deepcopy(X)
    y_shuffled = []
    X_shuffled = []
    if random_state != None:
        np.random.seed(random_state)
    if shuffle:
        for i in range(0, len(y_copy)):
            rand_idx = np.random.randint(low= 0, high= len(y_copy))
            y_shuffled.append(y_copy[rand_idx])
            y_copy.pop(rand_idx)
            X_shuffled.append(X_copy[rand_idx])
            X_copy.pop(rand_idx)
    else:
        X_shuffled = X_copy
        y_shuffled = y_copy

    for i in range(0, index_for_split):
        y_train.append(y_shuffled[i])
        X_train.append(X_shuffled[i])
    for i in range(index_for_split, len(X)):
        y_test.append(y_shuffled[i])
        X_test.append(X_shuffled[i])
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    folds = []
    X_copy = copy.deepcopy(X)
    X_shuffled = []
    X_indices2 = []

    if shuffle :
        if random_state != None:
            random.seed(random_state)
        for i in range(0, len(X)):
            rand_idx = random.randint(0,(len(X_copy)-1))
            X_shuffled.append(X_copy[rand_idx])
            X_indices2.append(rand_idx)
            X_copy.pop(rand_idx)
    else:
        for i in range(0, len(X)):
            X_indices2.append(i)

    X_indices = X_indices2.copy()
    n_samples = len(X)
    pre_fold = []
    training = []
    testing = []
    for i in range(0, n_samples % n_splits):
        size = n_samples//n_splits + 1
        for i in range(0, size):
            pre_fold.append(X_indices[0])
            X_indices.pop(0)
        training.append(pre_fold)
        pre_fold = []
    for i in range(n_samples % n_splits, n_splits):
        size = n_samples // n_splits
        for i in range(0, size):
            pre_fold.append(X_indices[0])
            X_indices.pop(0)
        training.append(pre_fold)
        pre_fold = []

    for i in range(0, len(training)):
        testing.append([])
        for val in X_indices2:
            if val not in training[i]:
                testing[i].append(val)

    for i in range(0, len(testing)):
        folds.append((testing[i], training[i]))
    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    labels = []
    indices = []
    training = []
    testing = []
    folds = []

    # creating a list of the classes
    for val in y:
        if val not in labels:
            labels.append(val)
    # creating a list of the indices based on their class label
    for j in range(0, len(labels)):
        indices.append([])
        for i in range(0, len(y)):
            if y[i] == labels[j]:
                indices[j].append(i)
    # creating and sizing the arrays of the folds
    for i in range(0, n_splits):
        testing.append([])
        training.append([])

    # if shuffle:
    #     shuffle_item = []
    #     if random_state != None:
    #         random.seed(random_state)
    #     for i in range(0, len(indices)):
    #         shuffle_item.append([])
    #         j = 0
    #         finished = False
    #         while not finished:
    #             rand_idx = random.randint(0,(len(indices[i])-1))
    #             shuffle_item[i].append(indices[i][rand_idx])
    #             indices[i].pop(rand_idx)
    #             j += 1
    #             if len(indices[i]) <= 0:
    #                 finished = True
    #     indices = shuffle_item

    # # appending the selected indices to the arrays
    # items_used = 0
    # item = []
    # counter = 1
    # for val in indices:
    #     item.append(0)
    # for j in range(0,   -(len(X)//-n_splits)):
    #     for i in range(0, n_splits):

    #         if items_used < len(X) and item[j%len(indices)] < len(indices[j%len(indices)]):
    #             testing[i].append(indices[j%len(indices)][item[j%len(indices)]])
    #             items_used += 1
    #             item[j%len(indices)] += 1
    #         elif items_used < len(X):
    #             testing[i].append(indices[(j%len(indices)) + counter][item[j%len(indices) + counter]])
    #             items_used += 1
    #             item[(j%len(indices)) + counter] += 1

    # for i in range(0, len(testing)):
    #     for j in range(0, len(indices)):
    #         for val in indices[j]:
    #             if val not in testing[i]:
    #                 training[i].append(val)
    # for i in range(0, len(testing)):
    #     folds.append((training[i], testing[i]))

    items_used = 0
    item = []
    for val in indices:
        item.append(0)

    curr_fold = 0
    for group in indices:
        for index in group:
            testing[curr_fold].append(index)
            curr_fold += 1
            curr_fold %= n_splits

    for i in range(0, len(testing)):
        for j in range(0, len(indices)):
            for val in indices[j]:
                if val not in testing[i]:
                    training[i].append(val)
    for i in range(0, len(testing)):
        folds.append((training[i], testing[i]))

    return folds

# def cross_val_score(clf, folds):
#     all_y_test = []
#     all_y_predicted = []
#     for train_indexes, test_indexes in folds:
#         X_train = [X_instances[i] for i in train_indexes]
#         y_train = [y_instances[i] for i in train_indexes]
#         X_test = [X_instances[i] for i in test_indexes]
#         y_test = [y_instances[i] for i in test_indexes]
#         all_y_test.extend(y_test)
#         clf.fit(X_train, y_train)
#         y_predicted = clf.predict(X_test)
#         all_y_predicted.extend(y_predicted)
#     print(myevaluation.accuracy_score(all_y_test, all_y_predicted))

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if y is None:
        y_input = False
    else:
        y_input = True
    y_sample = []
    X_sample = []
    X_out_of_bag = []
    y_out_of_bag = []
    if n_samples == None:
        n_samples = len(X)
    for val in range(0, n_samples):
        index = np.random.randint(0, len(X))
        X_sample.append(X[index])
        if y_input:
            y_sample.append(y[index])
    for val in X:
        if val not in X_sample:
            X_out_of_bag.append(val)
            if y_input:
                y_out_of_bag.append(y[X.index(val)])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for val in labels:
        matrix.append([])
    count = 0
    for val in labels:
        for val in labels:
            matrix[count].append(0)
        count += 1
    count = 0

    y_true_sorted = sorted(y_true)
    y_pred_sorted = [x for _,x in sorted(zip(y_true,y_pred))]

    for val in y_true_sorted:
        matrix[labels.index(val)][labels.index(y_pred_sorted[count])] += 1
        count += 1
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    percent_correct = 0.0
    number_correct = 0
    i = 0
    for val in y_pred:
        if val == y_true[i]:
            number_correct += 1
        i += 1
    percent_correct = number_correct/len(y_pred)
    if normalize:
        return percent_correct
    else:
        return number_correct


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    
    if labels == None:
        labels = []
        for val in y_true:
            if val not in labels:
                labels.append(val)
    if pos_label == None:
        pos_label = labels[0]

    num_correct = 0
    num_total = len(y_true)
    num_wrong = 0
    for i in range(0, len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == y_pred[i]:
                num_correct += 1
            else:
                num_wrong += 1
    if (num_correct + num_wrong) == 0:
        return 0.0
    else:
        precision = float(float(num_correct) /(num_correct + num_wrong))
        return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels == None:
        labels = []
        for val in y_true:
            if val not in labels:
                labels.append(val)
    if pos_label == None:
        pos_label = labels[0]

    num_correct = 0
    num_total = len(y_true)
    num_wrong = 0
    for i in range(0, len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == y_pred[i]:
                num_correct += 1
        elif(y_pred[i] != pos_label):
            if y_true[i] == pos_label:
                num_wrong += 1
    if (num_correct + num_wrong) == 0:
        return 0.0
    else:
        recall = float(float(num_correct) /(num_correct + num_wrong))
        return recall
    

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if ((binary_recall_score(y_true, y_pred, labels, pos_label) + binary_precision_score(y_true, y_pred, labels, pos_label)) == 0):
        return 0.0
    else:
        F1 = 2 * ((binary_recall_score(y_true, y_pred, labels, pos_label) * binary_precision_score(y_true, y_pred, labels, pos_label))/ (binary_recall_score(y_true, y_pred, labels, pos_label) + binary_precision_score(y_true, y_pred, labels, pos_label)))
        return F1

