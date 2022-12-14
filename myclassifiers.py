from mysklearn import myutils as u
import numpy as np
from mysklearn import myevaluation
class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        X_train = [x[0] for x in X_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
            y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        predictions = []
        if self.slope is not None and self.intercept is not None:
            for test_instance in X_test:
                item = self.slope * test_instance[0] + self.intercept
                predictions.append(item)
        return predictions

    @staticmethod # decorator to denote this is a static (class-level) method
    def compute_slope_intercept(x, y):
        """Fits a simple univariate line y = mx + b to the provided x y data.
        Follows the least squares approach for simple linear regression.

        Args:
            x(list of numeric vals): The list of x values
            y(list of numeric vals): The list of y values

        Returns:
            m(float): The slope of the line fit to x and y
            b(float): The intercept of the line fit to x and y
        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
            / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
        # y = mx + b => y - mx
        b = mean_y - m * mean_x
        return m, b

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted_numeric = self.regressor.predict(X_test)
        y_predicted = []
        for pred in y_predicted_numeric:
            y_predicted.append(self.discretizer(pred))
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        row_indexes_dists = []
        counter = 0
        for test_instance in X_test:
            distances.append([])
            neighbor_indices.append([])
            for i, train_instance in enumerate(self.X_train):
                dist = u.dist_for_categorical(train_instance, test_instance)
                row_indexes_dists.append((i, dist))
            for val in row_indexes_dists:
                distances[counter].append(val[1])
                neighbor_indices[counter].append(val[0])
            counter += 1

        return distances, neighbor_indices 

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predict_array = []
        
        distances, neighbor_indices = self.kneighbors(X_test)
        count = 0
        for val in distances:
            potential_predict_list = []
            counter_array = []
            sorted_dist, sorted_indices = u.parallel_sort(val, neighbor_indices[count])
            top_k_dist = sorted_dist[:self.n_neighbors]
            top_k_indices = sorted_indices[:self.n_neighbors]
            for item in top_k_indices:
                if self.y_train[count] not in potential_predict_list:
                    potential_predict_list.append(item)
                    counter_array.append(1)
                else:
                    counter_array[potential_predict_list.index(item)] += 1
            predict_array.append(self.y_train[potential_predict_list[max(counter_array)]])
            count += 1
        return predict_array

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        counts = []
        items = []
        for val in y_train:
            if val not in items:
                items.append(val)
                counts.append(0)
        for val in y_train:
            for item in items:
                if item == val:
                    counts[items.index(item)] += 1

        self.most_common_label = items[counts.index(max(counts))]


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for val in X_test:
            y_predicted.append(self.most_common_label)
        return y_predicted
class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None #class label
        self.posteriors = None
        self.class_labels = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        priors = {}
        instance_count = len(X_train)
        classes = []
        class_count = {}
        for val in y_train:
            if val not in classes:
                classes.append(val)
        self.class_labels = classes
        for val in classes:
            class_count[val] = 0
            i = 0
            for item in y_train:
                if val == item:
                    i += 1
                    class_count[val] += 1
                    priors[val] = (i, instance_count)
        self.priors = priors
    
        # posteriors
        posteriors = {}
        for val in classes:
            run = False
            j = 0
            posteriors[val] = {}
            for count in range(1, len(X_train[0]) +1):
                
                attributes = []
                attribute_count = []
                
                for i in range(0, len(X_train)):
                    if X_train[i][j] not in attributes:
                        attributes.append(X_train[i][j])
                        attribute_count.append(0)
                    if y_train[i] == val:
                        attribute_count[attributes.index(X_train[i][j])] += 1
                j += 1   
    
                if not run:
                    i = 1
                    for item in X_train[0]:
                        posteriors[val]["att" + str(i)] = {}
                        i += 1
                        run = True
                
                for thing in attributes:
                    posteriors[val]["att" + str(count)][thing] = (attribute_count[attributes.index(thing)],class_count[val])
                
        self.posteriors = posteriors



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        # {"yes":(5, 8), "no":(3, 8)}
        # unseen_instance = [[2, 2, "fair"], [1, 1, "excellent"]]
        #  {"yes": {"att1": {1: (4,5), 2: (1,5)}, "att2":{5:(2,5), 6: (3,5)}}, 
        #   "no": {"att1": {1: (2,3), 2: (1,3)}, "att2":{5:(2,3), 6: (1,3)}}}
        for instance in X_test:
            calc_vals = []
            for val in self.class_labels:
                calc_prob = self.priors[val][0]/self.priors[val][1]
                for i in range(0, len(self.posteriors[val])):
                    calc_prob *= (self.posteriors[val]["att" + str(i + 1)][instance[i]][0]/self.posteriors[val]["att" + str(i + 1)][instance[i]][1])
                calc_vals.append(calc_prob)
            y_predicted.append(self.class_labels[calc_vals.index(max(calc_vals))])
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.tree = None

    def fit(self, X_train, y_train, F):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # BUILDING HEADER AND ATTRIBUTE DOMAINS
        header = []
        attribute_domains = {}
        test = len(X_train[0])
        for i in range(0, len(X_train[0])):
            header.append("att" + str(i))
            attribute_domains["att" + str(i)] = []
        for i in range(0, len(header)):
            for j in range(0, len(X_train)):
                if X_train[j][i] not in attribute_domains["att" + str(i)]:
                    attribute_domains["att" + str(i)].append(X_train[j][i])
        #starter code
        # # next recommended that x_train and y_train stitch together, so class label is at instance[-1]
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = header.copy() # recall tdidt is going to be removin attributes from a list of available attributes
        # # python is pass by object reference!!
        self.tree = u.tdidt(train, available_attributes, attribute_domains, -1, header, F)
        self.header = header
        

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            prediction = u.tdidt_predict(self.tree, instance, self.header)
            y_predicted.append(prediction) 
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names == None:
            attribute_names = self.header
        u.print_tree(self.tree, self.header, attribute_names, class_name, "IF ")

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
class MyRandomForestClassifier:

    def __init__(self):
        """Initializer for MyRandomForestClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.F = None
        self.N = None
        self.M = None
        self.tree_list = []


    def fit(self, table, N, M, F):

        self.N = N
        self.M = M
        self.F = F

        X, y = u.train_splits(table)

        self.X_train = X
        self.y_train = y

        
        training_sample, validation_sample = u.compute_bootstrapped_sample(table)
        
        X, y = u.train_splits(training_sample)
        full_forest = u.forest(X, y, N, F)

        X_test, y_test = u.train_splits(validation_sample)

        averages = []
        results = {}
        for i, tree in enumerate(full_forest):
            tree_predicted = tree.predict(X_test)
            results[i] = []
            results[i].append(
                myevaluation.accuracy_score(y_test, tree_predicted))
            results[i].append(
                myevaluation.binary_precision_score(y_test, tree_predicted))
            results[i].append(
                myevaluation.binary_recall_score(y_test, tree_predicted))
            results[i].append(
                myevaluation.binary_f1_score(y_test, tree_predicted))
            averages.append([i, sum(results[i]) / len(results[i])])
        sorted = averages.copy()
        sorted.sort(key=lambda x: x[1], reverse=True)
        best_trees = sorted[:M]
        final_forest = []
        for i in best_trees:
            final_forest.append(full_forest[i[0]])
        self.tree_list = final_forest

    def predict(self, X_test, y_test):
        
        y_pred = []
        tree_accuracy = []
        for tree in self.tree_list:
            predicted = tree.predict(X_test)
            y_pred.append(predicted)
            tree_accuracy.append(myevaluation.accuracy_score(y_test, predicted))

        # for row in X_test:           
        #     predictions = []
        #     for tree in self.tree_list:
        #         predictions.append(u.tdidt_predict(tree.tree, row, tree.header))

        return y_pred[tree_accuracy.index(max(tree_accuracy))]