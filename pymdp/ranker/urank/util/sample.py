import tensorflow as tf
from util import loss_fns

def max_label_sample(labels, predicted_scores):
	permutation_loss = loss_fns.get_listnet_loss(labels, predicted_scores)

	predicted_scores = tf.reshape(predicted_scores, [-1, 1])
	labels = tf.reshape(labels, [-1, 1])
	n_data = tf.shape(labels)[0]
	# # [[1. 0. 0.]]
	transposed_labels = tf.transpose(labels)
	exps = tf.exp(transposed_labels)
	# # keep it substruct 
	exp_sum = tf.reduce_sum(exps)
	# # [[0.5761169  0.21194157 0.21194157]]
	softmax_scores = tf.divide(exps, exp_sum)
	# # [[1]]
	action_0_index = tf.multinomial(softmax_scores, 1)
	action_0_index = tf.cast(action_0_index, tf.int32)
	# predicted_scores has been vertical
	action_0 = tf.gather(predicted_scores, tf.reshape(action_0_index, [-1, 1]))
	action_0 = tf.reshape(action_0, [1, -1])
	actions = action_0

	label_0 = tf.gather(labels, tf.reshape(action_0_index, [-1, 1]))
	label_0 = tf.reshape(label_0, [1, -1])
	new_labels = label_0


	indices = tf.range(n_data, dtype=tf.int32)
	keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_0_index)))
	# predicted_scores have always been vertical
	# keep_predicted_scores have always been vertical	
	keep_predicted_scores = tf.gather_nd(predicted_scores, keep_indices)
	
	keep_labels = tf.gather_nd(labels, keep_indices)
	# horizontal label
	keep_labels = tf.transpose(keep_labels)

	log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))

	iteration = n_data - 1
	shape_invariants = [iteration.get_shape(), tf.TensorShape([None, 1]), 
	tf.TensorShape([None, 1]), tf.TensorShape([1, None]), tf.TensorShape([None, 1]), permutation_loss.get_shape()]

	def _cond(iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss):
	    return tf.greater(iteration, 1)

	def _gen_loop_body():
	    def loop_body(iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss):
	        discount = tf.log(tf.cast(tf.squeeze(n_data - iteration) + 2, dtype=tf.float32)) / log_2

	        exps = tf.exp(keep_labels)
	        # # keep it substruct 
	        exp_sum = tf.reduce_sum(exps)
	        # # [[0.5761169  0.21194157 0.21194157]]
	        softmax_scores = tf.divide(exps, exp_sum)
	        # # [[1]]
	        action_i_index = tf.multinomial(softmax_scores, 1)
	        action_i_index = tf.cast(action_i_index, tf.int32)
	        action_i = tf.gather(keep_predicted_scores, tf.reshape(action_i_index, [-1, 1]))
	        action_i = tf.reshape(action_i, [1, -1])
	        actions = tf.concat([actions, action_i], 0)
	        # similar for labels
	        label_i = tf.gather(tf.transpose(keep_labels), tf.reshape(action_i_index, [-1, 1]))
	        label_i = tf.reshape(label_i, [1, -1])
	        new_labels = tf.concat([new_labels, label_i], 0)
	        
	        indices = tf.range(iteration, dtype=tf.int32)
	        keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_i_index)))
	        keep_predicted_scores = tf.gather_nd(keep_predicted_scores, keep_indices)
	        
	        # labels have always been vertical
	        # keep_labels have always been vertical
	        keep_labels = tf.gather_nd(tf.transpose(keep_labels), keep_indices)
	        keep_labels = tf.transpose(keep_labels) 
	        
	        return tf.subtract(iteration, 1), keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss
	    return loop_body

	iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss = tf.while_loop(_cond, _gen_loop_body(),
	[iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss], shape_invariants=shape_invariants)	
	# the final action
	actions = tf.concat([actions, keep_predicted_scores], 0)
	new_labels = tf.concat([new_labels, keep_labels], 0)
	new_labels = tf.reshape(new_labels, [-1, 1])
	actions = tf.reshape(actions, [-1, 1])	
	return 	new_labels, actions, permutation_loss

def softmax_label_sample(labels, predicted_scores):
	permutation_loss = loss_fns.get_listnet_loss(labels, predicted_scores)

	predicted_scores = tf.reshape(predicted_scores, [-1, 1])
	labels = tf.reshape(labels, [-1, 1])
	n_data = tf.shape(labels)[0]
	# # [[1. 0. 0.]]
	transposed_labels = tf.transpose(labels)
	exps = tf.exp(transposed_labels)
	# # keep it substruct 
	exp_sum = tf.reduce_sum(exps)
	# # [[0.5761169  0.21194157 0.21194157]]
	softmax_scores = tf.divide(exps, exp_sum)
	# # [[1]]
	action_0_index = tf.multinomial(softmax_scores, 1)
	action_0_index = tf.cast(action_0_index, tf.int32)
	# predicted_scores has been vertical
	action_0 = tf.gather(predicted_scores, tf.reshape(action_0_index, [-1, 1]))
	action_0 = tf.reshape(action_0, [1, -1])
	actions = action_0

	label_0 = tf.gather(labels, tf.reshape(action_0_index, [-1, 1]))
	label_0 = tf.reshape(label_0, [1, -1])
	new_labels = label_0


	indices = tf.range(n_data, dtype=tf.int32)
	keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_0_index)))
	# predicted_scores have always been vertical
	# keep_predicted_scores have always been vertical	
	keep_predicted_scores = tf.gather_nd(predicted_scores, keep_indices)
	
	keep_labels = tf.gather_nd(labels, keep_indices)
	# horizontal label
	keep_labels = tf.transpose(keep_labels)

	log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))

	iteration = n_data - 1
	shape_invariants = [iteration.get_shape(), tf.TensorShape([None, 1]), 
	tf.TensorShape([None, 1]), tf.TensorShape([1, None]), tf.TensorShape([None, 1]), permutation_loss.get_shape()]

	def _cond(iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss):
	    return tf.greater(iteration, 1)

	def _gen_loop_body():
	    def loop_body(iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss):
	        discount = tf.log(tf.cast(tf.squeeze(n_data - iteration) + 2, dtype=tf.float32)) / log_2
	        exps = tf.exp(keep_labels)
	        # # keep it substruct 
	        exp_sum = tf.reduce_sum(exps)
	        # # [[0.5761169  0.21194157 0.21194157]]
	        softmax_scores = tf.divide(exps, exp_sum)
	        # # [[1]]
	        action_i_index = tf.multinomial(softmax_scores, 1)
	        action_i_index = tf.cast(action_i_index, tf.int32)
	        action_i = tf.gather(keep_predicted_scores, tf.reshape(action_i_index, [-1, 1]))
	        action_i = tf.reshape(action_i, [1, -1])
	        actions = tf.concat([actions, action_i], 0)
	        # similar for labels
	        label_i = tf.gather(tf.transpose(keep_labels), tf.reshape(action_i_index, [-1, 1]))
	        label_i = tf.reshape(label_i, [1, -1])
	        new_labels = tf.concat([new_labels, label_i], 0)
	        
	        indices = tf.range(iteration, dtype=tf.int32)
	        keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_i_index)))
	        keep_predicted_scores = tf.gather_nd(keep_predicted_scores, keep_indices)
	        
	        # labels have always been vertical
	        # keep_labels have always been vertical
	        keep_labels = tf.gather_nd(tf.transpose(keep_labels), keep_indices)
	        keep_labels = tf.transpose(keep_labels) 
	        
	        return tf.subtract(iteration, 1), keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss
	    return loop_body

	iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss = tf.while_loop(_cond, _gen_loop_body(),
	[iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss], shape_invariants=shape_invariants)	
	# the final action
	actions = tf.concat([actions, keep_predicted_scores], 0)
	new_labels = tf.concat([new_labels, keep_labels], 0)
	new_labels = tf.reshape(new_labels, [-1, 1])
	actions = tf.reshape(actions, [-1, 1])	
	return 	new_labels, actions, permutation_loss


def get_max_actions(labels, predicted_scores):
	permutation_loss = loss_fns.get_listnet_loss(labels, predicted_scores)

	predicted_scores = tf.reshape(predicted_scores, [-1, 1])
	labels = tf.reshape(labels, [-1, 1])
	n_data = tf.shape(labels)[0]
	# # [[1. 0. 0.]]
	transposed_predicted_scores = tf.transpose(predicted_scores)
	# exp_squeezed_predicted_scores = tf.exp(predicted_scores_variable)
	exps = tf.exp(transposed_predicted_scores)
	# # keep it substruct 
	exp_sum = tf.reduce_sum(exps)
	# # [[0.5761169  0.21194157 0.21194157]]
	softmax_scores = tf.divide(exps, exp_sum)
	# # [[1]]
	# action_0_index = tf.multinomial(softmax_scores, 1)
	action_0_index = tf.argmax(softmax_scores, axis=1)
	action_0_index = tf.cast(action_0_index, tf.int32)
	action_0 = tf.gather(tf.transpose(transposed_predicted_scores), tf.reshape(action_0_index, [-1, 1]))
	action_0 = tf.reshape(action_0, [1, -1])
	actions = action_0
	# labels have always been vertical
	label_0 = tf.gather(labels, tf.reshape(action_0_index, [-1, 1]))
	label_0 = tf.reshape(label_0, [1, -1])
	new_labels = label_0

	log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))

	indices = tf.range(n_data, dtype=tf.int32)
	keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_0_index)))
	keep_predicted_scores = tf.gather_nd(tf.transpose(transposed_predicted_scores), keep_indices)
	keep_predicted_scores = tf.transpose(keep_predicted_scores)
	# labels have always been vertical
	# keep_labels have always been vertical
	keep_labels = tf.gather_nd(labels, keep_indices)


	iteration = n_data - 1
	shape_invariants = [iteration.get_shape(), tf.TensorShape([1, None]), 
	tf.TensorShape([None, 1]), tf.TensorShape([None, 1]), tf.TensorShape([None, 1]), permutation_loss.get_shape()]

	def _cond(iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss):
	    return tf.greater(iteration, 1)

	def _gen_loop_body():
	    def loop_body(iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss):
	        discount = tf.log(tf.cast(tf.squeeze(n_data - iteration) + 2, dtype=tf.float32)) / log_2
	        exps = tf.exp(keep_predicted_scores)
	        # # keep it substruct 
	        exp_sum = tf.reduce_sum(exps)
	        # # [[0.5761169  0.21194157 0.21194157]]
	        softmax_scores = tf.divide(exps, exp_sum)
	        # # [[1]]
	        # action_i_index = tf.multinomial(softmax_scores, 1)
	        action_i_index = tf.argmax(softmax_scores, axis=1)
	        action_i_index = tf.cast(action_i_index, tf.int32)
	        action_i = tf.gather(tf.transpose(keep_predicted_scores), tf.reshape(action_i_index, [-1, 1]))
	        action_i = tf.reshape(action_i, [1, -1])
	        actions = tf.concat([actions, action_i], 0)
	        # similar for labels
	        label_i = tf.gather(keep_labels, tf.reshape(action_i_index, [-1, 1]))
	        label_i = tf.reshape(label_i, [1, -1])
	        new_labels = tf.concat([new_labels, label_i], 0)
	        
	        indices = tf.range(iteration, dtype=tf.int32)
	        keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_i_index)))
	        vertical_keep_tensor = tf.gather_nd(tf.transpose(keep_predicted_scores), keep_indices)
	        keep_predicted_scores = tf.transpose(vertical_keep_tensor)
	        
	        # labels have always been vertical
	        # keep_labels have always been vertical
	        keep_labels = tf.gather_nd(keep_labels, keep_indices) 
	        
	        return tf.subtract(iteration, 1), keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss
	    return loop_body

	iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss = tf.while_loop(_cond, _gen_loop_body(),
	[iteration, keep_predicted_scores, actions, keep_labels, new_labels, permutation_loss], shape_invariants=shape_invariants)	
	# the final action
	actions = tf.concat([actions, keep_predicted_scores], 0)
	new_labels = tf.concat([new_labels, keep_labels], 0)
	new_labels = tf.reshape(new_labels, [-1, 1])
	actions = tf.reshape(actions, [-1, 1])
	return 	new_labels, actions, permutation_loss


def softmax_sample(labels, predicted_scores):

	predicted_scores = tf.reshape(predicted_scores, [-1, 1])
	labels = tf.reshape(labels, [-1, 1])
	n_data = tf.shape(labels)[0]
	# # [[1. 0. 0.]]
	transposed_predicted_scores = tf.transpose(predicted_scores)
	exps = tf.exp(transposed_predicted_scores)
	# # keep it substruct 
	exp_sum = tf.reduce_sum(exps)
	# # [[0.5761169  0.21194157 0.21194157]]
	softmax_scores = tf.divide(exps, exp_sum)
	# # [[1]]
	action_0_index = tf.multinomial(softmax_scores, 1)
	action_0_index = tf.cast(action_0_index, tf.int32)
	action_0 = tf.gather(tf.transpose(transposed_predicted_scores), tf.reshape(action_0_index, [-1, 1]))
	action_0 = tf.reshape(action_0, [1, -1])
	actions = action_0
	# labels have always been vertical
	label_0 = tf.gather(labels, tf.reshape(action_0_index, [-1, 1]))
	label_0 = tf.reshape(label_0, [1, -1])
	new_labels = label_0


	indices = tf.range(n_data, dtype=tf.int32)
	keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_0_index)))
	keep_predicted_scores = tf.gather_nd(tf.transpose(transposed_predicted_scores), keep_indices)
	keep_predicted_scores = tf.transpose(keep_predicted_scores)
	# labels have always been vertical
	# keep_labels have always been vertical
	keep_labels = tf.gather_nd(labels, keep_indices)

	iteration = n_data - 1
	shape_invariants = [iteration.get_shape(), tf.TensorShape([1, None]), 
	tf.TensorShape([None, 1]), tf.TensorShape([None, 1]), tf.TensorShape([None, 1])]

	def _cond(iteration, keep_predicted_scores, actions, keep_labels, new_labels):
	    return tf.greater(iteration, 1)

	def _gen_loop_body():
	    def loop_body(iteration, keep_predicted_scores, actions, keep_labels, new_labels):

	        exps = tf.exp(keep_predicted_scores)
	        # # keep it substruct 
	        exp_sum = tf.reduce_sum(exps)
	        # # [[0.5761169  0.21194157 0.21194157]]
	        softmax_scores = tf.divide(exps, exp_sum)
	        # # [[1]]
	        action_i_index = tf.multinomial(softmax_scores, 1)
	        action_i_index = tf.cast(action_i_index, tf.int32)
	        action_i = tf.gather(tf.transpose(keep_predicted_scores), tf.reshape(action_i_index, [-1, 1]))
	        action_i = tf.reshape(action_i, [1, -1])
	        actions = tf.concat([actions, action_i], 0)
	        # similar for labels
	        label_i = tf.gather(keep_labels, tf.reshape(action_i_index, [-1, 1]))
	        label_i = tf.reshape(label_i, [1, -1])
	        new_labels = tf.concat([new_labels, label_i], 0)
	        
	        indices = tf.range(iteration, dtype=tf.int32)
	        keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_i_index)))
	        vertical_keep_tensor = tf.gather_nd(tf.transpose(keep_predicted_scores), keep_indices)
	        keep_predicted_scores = tf.transpose(vertical_keep_tensor)
	        
	        # labels have always been vertical
	        # keep_labels have always been vertical
	        keep_labels = tf.gather_nd(keep_labels, keep_indices) 
	        
	        return tf.subtract(iteration, 1), keep_predicted_scores, actions, keep_labels, new_labels
	    return loop_body

	iteration, keep_predicted_scores, actions, keep_labels, new_labels = tf.while_loop(_cond, _gen_loop_body(),
	[iteration, keep_predicted_scores, actions, keep_labels, new_labels], shape_invariants=shape_invariants)	
	# the final action
	actions = tf.concat([actions, keep_predicted_scores], 0)
	new_labels = tf.concat([new_labels, keep_labels], 0)
	new_labels = tf.reshape(new_labels, [-1, 1])
	actions = tf.reshape(actions, [-1, 1])
	return new_labels, actions

def shuffle_docs(labels, features, height, width):
    n_data = tf.shape(labels)[0]
    indices = tf.range(n_data)
    labels_features = tf.concat([labels, features], 1)
    shuffled = tf.random_shuffle(labels_features)
    column_rows = tf.transpose(shuffled)

    new_labels = tf.gather(column_rows, [0])
    new_features = tf.gather(column_rows, tf.range(1, width + 1))
    # transpose back
    new_labels = tf.transpose(new_labels) # , [-1, 1]
    new_features = tf.transpose(new_features) # , [height, width]  
    return new_labels, new_features

def random_sample(labels, predicted_scores):
	n_data = tf.shape(labels)[0]
	indices = tf.range(n_data)
	labels_predicted_scores = tf.concat([labels, predicted_scores], 1)
	shuffled = tf.random_shuffle(labels_predicted_scores)
	column_rows = tf.transpose(shuffled)
	new_labels = tf.gather(column_rows, [0])
	new_logits = tf.gather(column_rows, [1])
	# transpose back
	new_labels = tf.transpose(new_labels) # , [-1, 1]
	new_logits = tf.transpose(new_logits) # , [-1, 1]	
	return new_labels, new_logits