import tensorflow as tf


def diag_mask(pairwise_label_scores):
	masks = tf.ones(tf.shape(pairwise_label_scores))
	n_data = tf.shape(pairwise_label_scores)[0]
	# line 8 -9 == line 10
	not_consider = tf.diag(tf.ones([n_data]))
	masks = tf.subtract(masks, not_consider)
	# masks = tf.matrix_band_part(masks, 0, -1)
	masks = tf.cast(masks, dtype=tf.float32)
	pair_count = tf.reduce_sum(masks)
	return masks, pair_count

def full_mask(pairwise_label_scores):
	masks = tf.ones(tf.shape(pairwise_label_scores))
    # line 19 == line 20
	not_consider = tf.less_equal(pairwise_label_scores, 0.5)
	# not_consider = tf.equal(pairwise_label_scores, 0.5)
	not_consider = tf.cast(not_consider, tf.float32)
	masks = tf.subtract(masks, not_consider)
	masks = tf.cast(masks, dtype=tf.float32)
	pair_count = tf.reduce_sum(masks)
	return masks, pair_count

def pruned_mask(pairwise_label_scores):
    #    0   1   2
    # 0  0, -1, -2
    # 1  1,  0, -1
    # 2  2,  1,  0 
    # ONLY KEEP THE POSITIVE DIFFS
    #    0   1   2
    # 0  0,  0,  0
    # 1  1,  0,  0
    # 2  2,  1,  0
    masks = tf.greater(pairwise_label_scores, 0.0)
    masks = tf.cast(masks, dtype=tf.float32)
    pair_count = tf.reduce_sum(masks)
    return masks, pair_count

def equal_mask(pairwise_label_scores):
    masks = tf.equal(pairwise_label_scores, 0)
    masks = tf.cast(masks, dtype=tf.float32)
    # take the uppder triangle (leave out diag)
    masks = tf.matrix_band_part(masks, 0, -1)    
    pair_count = tf.reduce_sum(masks)
    return masks, pair_count

def list_mask(raw_pairwise_label_scores):
    masks = tf.ones(tf.shape(raw_pairwise_label_scores))
    not_consider = tf.less_equal(raw_pairwise_label_scores, 0.0)
    not_consider = tf.cast(not_consider, tf.float32)
    masks = tf.subtract(masks, not_consider)
    masks = tf.cast(masks, dtype=tf.float32)
    return masks

def list_negative_mask(raw_pairwise_label_scores):
    masks = tf.ones(tf.shape(raw_pairwise_label_scores))
    not_consider = tf.greater_equal(raw_pairwise_label_scores, 0.0)
    not_consider = tf.cast(not_consider, tf.float32)
    masks = tf.subtract(masks, not_consider)
    masks = -tf.cast(masks, dtype=tf.float32)
    return masks
