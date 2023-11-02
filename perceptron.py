from supervisedlearner import SupervisedLearner
import numpy as np

class Perceptron(SupervisedLearner):
    def __init__(self, feature_funcs, lr, is_c):
        """

        :param lr: the rate at which the weights are modified at each iteration.
        :param is_c: True if the perceptron is for classification problems,
                     False if the perceptron is for regression problems.

        """

        super().__init__(feature_funcs)
        self.weights = None
        self.learning_rate = lr
        self._trained = False
        self.is_classifier = is_c

    # TODO: Implement the rest of this class!

    def step_function(self, inp):
        """

        :param inp: a real number
        :return: the predicted label produced by the given input

        Assigns a label of 1.0 to the datapoint if <w,x> is a positive quantity
        otherwise assigns label 0.0. Should only be called when self.is_classifier
        is True.
        """
        if self.is_classifier:
            if inp > 0:
                return 1.0
            else:
                return 0.0
        else:
            return inp

    def train(self, X, Y):
        """

        :param X: a 2D numpy array where each row represents a datapoint
        :param Y: a 1D numpy array where i'th element is the label of the corresponding datapoint in X
        :return:

        Does not return anything; only learns and stores as instance variable self.weights a 1D numpy
        array whose i'th element is the weight on the i'th feature.
        Do not forget to include the bias in your calculation.
        """
        new_col = np.ones(len(X))
        new_col = np.split(new_col, len(X))
        new_X = np.append(X, new_col, 1)
        self.weights = np.zeros(len(new_X[1]))

        i = 0
        while i < 1000:
            j = 0
            for x in new_X:
                if self.is_classifier:
                    p = self.step_function(np.dot(x, self.weights))
                    if p != Y[j]:
                        self.weights += self.learning_rate * (Y[j] - p) * x
                    j += 1
                else:
                    p = np.dot(x, self.weights)
                    self.weights += self.learning_rate * (Y[j] - p) * x
                    j += 1
            i += 1
        self._trained = True

    def predict(self, x):
        """
        :param x: a 1D numpy array representing a single datapoints
        :return:

        Given a data point x, produces the learner's estimate
        of f(x). Use self.weights and make sure to use self.step_function
        if self.is_classifier is True
        """
        new_x = np.append(x,1)
        if self.is_classifier:
            return self.step_function(np.dot(new_x, self.weights))
        else:
            return np.dot(new_x, self.weights)

    def evaluate(self, datapoints, labels):
        """

        :param datapoints: a 2D numpy array where each row represents a datapoint
        :param labels: a 1D numpy array where i'th element is the label of the corresponding datapoint in datapoints
        :return:

        If self.is_classifier is True, returns the fraction (between 0 and 1)
        of the given datapoints to which the method predict(.) assigns the correct label
        If self.is_classifier is False, returns the Mean Squared Error (MSE)
        between the labels and the predictions of their respective inputs (You
        do not have to calculate the R2 Score)
        """
        if self.is_classifier:
            i = 0
            j = 0
            for x in datapoints:
                if self.predict(x) == labels[j]:
                    i += 1
                j += 1
            return i/len(labels)
        else:
            i = 0
            j = 0
            for x in datapoints:
                se = pow((labels[i] - self.predict(x)), 2)
                i += 1
                j += se
            return j/len(datapoints)