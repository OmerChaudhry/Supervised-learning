from supervisedlearner import SupervisedLearner
import numpy as np

class KNNClassifier(SupervisedLearner):
    def __init__(self, feature_funcs, k):
        super(KNNClassifier, self).__init__(feature_funcs)
        self.k = k
        self._features_ = []
        self.anchor_l = []

    def train(self, anchor_points, anchor_labels):
        """
        :param anchor_points: a 2D numpy array, in which each row is
						      a datapoint, without its label, to be used
						      for one of the anchor points

		:param anchor_labels: a list in which the i'th element is the correct label
		                      of the i'th datapoint in anchor_points

		Does not return anything; simply stores anchor_labels and the
		_features_ of anchor_points.
		"""
        for x in anchor_points:
            self._features_.append(self.compute_features(x))
        self.anchor_l = anchor_labels

    def predict(self, x):
        """
        Given a single data point, x, represented as a 1D numpy array,
		predicts the class of x by taking a plurality vote among its k
		nearest neighbors in feature space. Resolves ties arbitrarily.

		The K nearest neighbors are determined based on Euclidean distance
		in _feature_ space (so be sure to compute the features of x).

		Returns the label of the class to which x is predicted to belong.
		"""
        # A list containing the Euclidean distance of x from another point y,
        # each element of which is in the form (distance, y index)
        # Get the k closest points to x and their labels
        # Note: max(set(x), key=x.count) returns the mode of a list x.
        i = 0
        ed_list = []
        x_features = self.compute_features(x)

        for y in self._features_:
            ed_list.append((np.linalg.norm(y - x_features), i))
            i += 1
        ed_list.sort(key = lambda y:y[0])
        # e.g. f_list = [(0.01, 2), (0.02, 4), (0.02,5)]
        f_list = ed_list[0:self.k]
        f_labels = []
        for _,i in f_list:
            f_labels.append(self.anchor_l[i])
        final_label = max(set(f_labels), key=f_labels.count)
        return final_label

        

    def evaluate(self, datapoints, labels):
        """
        :param datapoints: a 2D numpy array, in which each row is a datapoint.
		:param labels: a 1D numpy array, in which the i'th element is the
		               correct label of the i'th datapoint.

		Returns the fraction (between 0 and 1) of the given datapoints to which
		predict(.) assigns the correct label
		"""
        # Count the number of correct predictions and find the model accuracy
        i = 0
        j = 0
        for x in datapoints:
            if self.predict(x) == labels[j]:
                i += 1
            j += 1
        return i/len(labels) 
