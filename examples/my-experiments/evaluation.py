import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from typing import Optional
import pdb

class Evaluator:

	def __init__(self):
		pass

	def evaluate_clustering(self, labels_true, labels_pred):
		self.cluster_accuracy = self.__compute_cluster_accuracy(labels_true, labels_pred)[0]
		self.nmi = self.__compute_nmi(labels_true, labels_pred)
		self.ari = self.__compute_ari(labels_true, labels_pred)
		greedy_labels = self.__transform_clusters_to_labels(labels_true, labels_pred)
		self.purity = self.__compute_purity(labels_true, greedy_labels)

	def print_evaluation(self):
		print('ACC: {:.2f} PUR: {:.2f} NMI: {:.2f} ARI: {:.2f}'.format(self.cluster_accuracy, self.purity, self.nmi, self.ari))

	def __transform_clusters_to_labels(self, labels_true, labels_pred):
		greedy_labels = np.zeros(shape=labels_pred.shape, dtype=int)
		#pdb.set_trace()
		# Find the cluster ids (labels_true)
		c_ids = np.unique(labels_pred)

		# Dictionary to transform cluster label to real label
		dict_clusters_to_labels = dict()

		# For every cluster find the most frequent data label
		for c_id in c_ids:
			indexes_of_cluster_i = np.where(c_id == labels_pred)
			elements, frequency = np.unique(labels_true[indexes_of_cluster_i], return_counts=True)
			true_label_index = np.argmax(frequency)
			true_label = elements[true_label_index]
			dict_clusters_to_labels[c_id] = true_label

		# Change the cluster labels to real labels
		for i, element in enumerate(labels_pred):
			greedy_labels[i] = dict_clusters_to_labels[element]

		return greedy_labels

	def __compute_cluster_accuracy(self, labels_true, labels_pred, cluster_number: Optional[int] = None):
		"""
		Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
		determine reassignments.

		:param labels_true: list of true cluster numbers, an integer array 0-indexed
		:param labels_pred: list of predicted cluster numbers, an integer array 0-indexed
		:param cluster_number: number of clusters, if None then calculated from input
		:return: reassignment dictionary, clustering accuracy
		"""
		if cluster_number is None:
			# assume labels are 0-indexed
			cluster_number = (max(labels_pred.max(), labels_true.max()) + 1)
		
	
		count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
		for i in range(labels_pred.size):
			count_matrix[labels_pred[i], labels_true[i]] += 1

		row_ind, col_ind = linear_assignment(count_matrix.max() - count_matrix)
		reassignment = dict(zip(row_ind, col_ind))
		accuracy = count_matrix[row_ind, col_ind].sum() / labels_pred.size

		return accuracy, reassignment
		

	def __compute_purity(self, labels_true, labels_pred):
		return accuracy_score(labels_true, labels_pred)

	def __compute_nmi(self, labels_true, labels_pred):
		return normalized_mutual_info_score(labels_true, labels_pred)

	def __compute_ari(self, labels_true, labels_pred):
		return adjusted_rand_score(labels_true, labels_pred)

'''

def main():
	evaluator = Evaluator()
	labels = np.array([0, 1, 2, 3])
	clusters = np.array([0, 1, 2, 3])
	
	evaluator.evaluate_clustering(labels, clusters)
	evaluator.print_evaluation()

if __name__ == '__main__':
	main()

'''