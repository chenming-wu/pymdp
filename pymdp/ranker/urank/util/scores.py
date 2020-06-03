import tensorflow as tf

def get_pairwise_scores(predicted_scores):
	pairwise_predicted_scores = predicted_scores - tf.transpose(predicted_scores)
	return pairwise_predicted_scores


def get_pairwise_label_scores(labels):
	pairwise_label_scores = labels - tf.transpose(labels)
	differences_ij = tf.maximum(tf.minimum(1.0, pairwise_label_scores), -1.0)
	pairwise_label_scores = (1.0 / 2.0) * (1.0 + differences_ij)
	return pairwise_label_scores


def get_softmax_pairwise_scores(predicted_scores):
	exp_predicted_scores = 2 ** predicted_scores
	exp_predicted_scores = tf.divide(exp_predicted_scores, tf.reduce_sum(exp_predicted_scores))
	pairwise_predicted_scores = exp_predicted_scores - tf.transpose(exp_predicted_scores)
	return pairwise_predicted_scores
