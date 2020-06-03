import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def safe_div(numerator, denominator, name='safe_div'):
  """Computes a safe divide which returns 0 if the denominator is zero.
  This can be used for calculating ndcg with corner case of all candidate
  documents having ratings 0.
  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator`.
    name: An optional name for the returned op.

  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return array_ops.where(
      math_ops.equal(denominator, 0),
      array_ops.zeros_like(numerator),
      math_ops.div(numerator, denominator),
      name=name)

def cal_ndcg(label_scores, predicted_scores, top_ks=[1, 3, 5, 10], use_predicted_order=False):
  """
  Calculate NDCG score for top_k_int ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: An int or an int `Tensor`.
  Returns:
    a `Tensor` that holds DCG / IDCG.
  """
  max_prediction_size = array_ops.size(predicted_scores)
  max_top_k = max(top_ks)
  max_top_k = tf.minimum(max_prediction_size, max_top_k)

  sorted_labels, predicted_order, sorted_predictions = _get_ranking_orders(
    label_scores, predicted_scores, top_k_int=max_top_k, use_predicted_order=use_predicted_order)

  predicted_relevance = _get_relevance_scores(predicted_order)
  sorted_relevance = _get_relevance_scores(sorted_labels)

  cg_discounts = _get_cg_discount(max_top_k)
  dcgs = predicted_relevance / cg_discounts
  idcgs = sorted_relevance / cg_discounts
  cumsum_dcgs = tf.cumsum(dcgs, exclusive=False)
  cumsum_idcgs = tf.cumsum(idcgs, exclusive=False)
  # idcg is 0 if label_scores are all 0
  ndcgs = safe_div(cumsum_dcgs, cumsum_idcgs, 'ndcgs')
  k_indices = [tf.minimum(max_prediction_size, v) - 1 for v in top_ks]
  k_ndcgs = tf.gather(ndcgs, k_indices)
  # dcg = _dcg_idcg(predicted_relevance, cg_discount)
  # idcg = _dcg_idcg(sorted_relevance, cg_discount)
  # the ndcg score of the batch
  # # idcg is 0 if label_scores are all 0
  # ndcg = safe_div(dcg, idcg, 'one_ndcg')
  return k_ndcgs

def cal_swapped_ndcg(label_scores, predicted_scores, top_k_int):
  """
  Calculate swapped NDCG score in Lambda Rank for full/top k ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: An int or an int `Tensor`. 
  Returns:
    a `Tensor` that holds swapped NDCG by .
  """
  sorted_labels, predicted_order, sorted_predictions = _get_ranking_orders(
    label_scores, predicted_scores, top_k_int=top_k_int)

  predicted_relevance = _get_relevance_scores(predicted_order)
  sorted_relevance = _get_relevance_scores(sorted_labels)

  cg_discount = _get_cg_discount(top_k_int)

  # cg_discount is safe as a denominator
  dcg_k = predicted_relevance / cg_discount
  dcg = tf.reduce_sum(dcg_k)

  idcg_k = sorted_relevance / cg_discount
  idcg = tf.reduce_sum(idcg_k)

  ndcg = safe_div(dcg, idcg, 'ndcg_in_lambdarank_training')

  # remove the gain from label i then add the gain from label j
  tiled_ij = tf.tile(dcg_k, [1, top_k_int])
  new_ij = (predicted_relevance / tf.transpose(cg_discount))

  tiled_ji = tf.tile(tf.transpose(dcg_k), [top_k_int, 1])
  new_ji = tf.transpose(predicted_relevance) / cg_discount

  # if swap i and j, remove the stale cg for i, then add the new cg for i,
  # remove the stale cg for j, and then add the new cg for j
  new_dcg = dcg - tiled_ij + new_ij - tiled_ji + new_ji

  new_ndcg = safe_div(new_dcg, idcg, 'new_ndcg_in_lambdarank_training')
  swapped_ndcg = tf.abs(ndcg - new_ndcg)
  return swapped_ndcg

def diff_idcg_dcg(label_scores, predicted_scores, top_k_int):
  return cal_idcg(label_scores, predicted_scores, top_k_int) - \
  cal_dcg(label_scores, predicted_scores, top_k_int)

def cal_idcg(label_scores, predicted_scores, top_k_int):
  """
  Calculate swapped NDCG score in Lambda Rank for full/top k ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: An int or an int `Tensor`. 
  Returns:
    a `Tensor` that holds swapped NDCG by .
  """
  max_prediction_size = array_ops.size(predicted_scores)
  max_top_k = tf.minimum(max_prediction_size, top_k_int)

  sorted_labels, predicted_order, sorted_predictions = _get_ranking_orders(
    label_scores, predicted_scores, top_k_int=max_top_k)

  sorted_relevance = _get_relevance_scores(sorted_labels)

  cg_discount = _get_cg_discount(max_top_k)

  # cg_discount is safe as a denominator
  idcg_ks = sorted_relevance / cg_discount

  return tf.reduce_sum(idcg_ks)


def cal_dcg(label_scores, predicted_scores, top_k_int):
  """
  Calculate swapped NDCG score in Lambda Rank for full/top k ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: An int or an int `Tensor`. 
  Returns:
    a `Tensor` that holds swapped NDCG by .
  """
  sorted_labels, predicted_order, sorted_predictions = _get_ranking_orders(
    label_scores, predicted_scores, top_k_int=top_k_int)

  predicted_relevance = _get_relevance_scores(predicted_order)

  cg_discount = _get_cg_discount(top_k_int)

  # cg_discount is safe as a denominator
  dcg_ks = predicted_relevance / cg_discount

  return tf.reduce_sum(dcg_ks)


def cal_dcg_ks(label_scores, top_k_int):
  """
  Calculate swapped NDCG score in Lambda Rank for full/top k ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: An int or an int `Tensor`. 
  Returns:
    a `Tensor` that holds swapped NDCG by .
  """

  predicted_relevance = _get_relevance_scores(label_scores)

  cg_discount = _get_cg_discount(top_k_int)

  # cg_discount is safe as a denominator
  dcg_k = predicted_relevance / cg_discount

  return dcg_k


def cal_idcg_ks(label_scores, top_k_int):
  """
  Calculate swapped NDCG score in Lambda Rank for full/top k ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: An int or an int `Tensor`. 
  Returns:
    a `Tensor` that holds swapped NDCG by .
  """
  # sorted_labels contians the relevance scores of the correct order
  sorted_labels, ordered_labels_indices = tf.nn.top_k(
    tf.transpose(label_scores), k=top_k_int)
  sorted_labels = tf.transpose(sorted_labels)  

  ideal_relevance = _get_relevance_scores(sorted_labels)

  cg_discount = _get_cg_discount(top_k_int)

  # cg_discount is safe as a denominator
  idcg_k = ideal_relevance / cg_discount

  return idcg_k

def cal_err(label_scores, predicted_scores, top_k_int=1, use_predicted_order=False):
  """
  Calculate NDCG score for top_k_int ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: An int or an int `Tensor`.
  Returns:
    a `Tensor` that holds DCG / IDCG.
  """
    
  if not use_predicted_order:
    max_label_score = tf.reduce_max(label_scores)
    sorted_labels, predicted_order, sorted_predictions = _get_ranking_orders(
      label_scores, predicted_scores, top_k_int=top_k_int)
  else:
    indices = tf.range(top_k_int)
    predicted_order = tf.gather(predicted_scores, indices)
    max_label_score = tf.reduce_max(label_scores)

  predicted_relevance = _get_relevance_scores(predicted_order)
  predicted_relevance_ratio = predicted_relevance / (2**max_label_score)
  _ratio = 1- predicted_relevance_ratio
  prob_stepdown = tf.cumprod(_ratio, exclusive=True)
  cur_rank = tf.range(top_k_int) + 1
  cur_rank = tf.cast(cur_rank, dtype=tf.float32)
  cur_rank = tf.reshape(cur_rank, [-1, 1])
  products = tf.multiply(predicted_relevance_ratio, prob_stepdown)
  products = tf.cast(products, dtype=tf.float32)
  errs = tf.divide(products, cur_rank)
  err = tf.reduce_sum(errs)
  return err


def _dcg_idcg(relevance_scores, cg_discount):
  """
  Calculate DCG scores for top_k_int ranking positions
  Args:
    relevance_scores: a real `Tensor`.
    cg_discount: a real `Tensor`, with dtype matching relevance_scores
  Returns:
    a `Tensor` that holds \sum_{i=1}^k \frac{relevance_scores_k}{cg_discount}  
  """
  # cg_discount is safe
  dcg_k = relevance_scores / cg_discount
  return tf.reduce_sum(dcg_k)


def _get_ranking_orders(label_scores, predicted_scores, top_k_int=1, use_predicted_order=False):
  """
  Calculate DCG scores for top_k_int ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: an integer or an int `Tensor`.
  Returns:
    two `Tensors` that hold sorted_labels: the ground truth relevance socres
    and predicted_order: relevance socres based on sorted predicted_scores
  """
  # sort predictions_scores and label_scores
  # size [batch_size/num of DataRecords, 1]
  # label_scores = tf.Print(label_scores, [label_scores], 'label_scores: \n', summarize=200)
  # predicted_scores = tf.Print(predicted_scores, [predicted_scores], 'predicted_scores: \n', summarize=200)
  predicted_scores = tf.reshape(predicted_scores, [-1, 1])
  label_scores = tf.reshape(label_scores, [-1, 1])
  if not use_predicted_order:
    # sort predicitons and use the indices to obtain the relevance scores of the predicted order
    sorted_predictions, ordered_predictions_indices = tf.nn.top_k(
      tf.transpose(predicted_scores), k=top_k_int)
    ordered_predictions_indices_for_labels = tf.transpose(ordered_predictions_indices)
    # predicted_order contians the relevance scores of the predicted order
    predicted_order = tf.gather_nd(label_scores, ordered_predictions_indices_for_labels)

  # !!!!! actions sudo predicted_scores (descending)
  else:
    indices = tf.range(top_k_int)
    predicted_order = tf.gather(predicted_scores, indices)
    sorted_predictions = tf.gather(predicted_scores, indices)
    # label_scores = tf.reshape(glrank_complete_labels, [-1, 1])
  
  # sorted_labels contians the relevance scores of the correct order
  sorted_labels, ordered_labels_indices = tf.nn.top_k(
    tf.transpose(label_scores), k=top_k_int)
  sorted_labels = tf.transpose(sorted_labels)

  # sorted_labels = tf.Print(sorted_labels, [sorted_labels], 'sorted_labels: \n', summarize=200)
  # predicted_order =  tf.Print(predicted_order, [predicted_order], 'predicted_order: \n', summarize=200)
  return sorted_labels, predicted_order, sorted_predictions


def get_logit_orders(label_scores, predicted_scores):
  """
  Calculate DCG scores for top_k_int ranking positions
  Args:
    label_scores: a real `Tensor`.
    predicted_scores: a real `Tensor`, with dtype matching label_scores
    top_k_int: an integer or an int `Tensor`.
  Returns:
    two `Tensors` that hold sorted_labels: the ground truth relevance socres
    and predicted_order: relevance socres based on sorted predicted_scores
  """
  # sort predictions_scores and label_scores
  # size [batch_size/num of DataRecords, 1]
  top_k_int = tf.shape(label_scores)[0]
  label_scores = tf.reshape(label_scores, [-1, 1])
  predicted_scores = tf.reshape(predicted_scores, [-1, 1])

  # sort predicitons and use the indices to obtain the relevance scores of the predicted order
  sorted_predictions, ordered_predictions_indices = tf.nn.top_k(
    tf.transpose(predicted_scores), k=top_k_int)
  ordered_predictions_indices_for_labels = tf.transpose(ordered_predictions_indices)
  # predicted_order contians the relevance scores of the predicted order
  predicted_order = tf.gather_nd(label_scores, ordered_predictions_indices_for_labels)
  sorted_predictions = tf.reshape(sorted_predictions, [-1, 1])
  return predicted_order, sorted_predictions


def _get_cg_discount(top_k_int=1):
  """
  Calculate discounted gain factor for ranking position till top_k_int
  Args:
    top_k_int: An int or an int `Tensor`.
  Returns:
    a `Tensor` that holds \log_{2}(i + 1), i \in [1, k] 
  """
  log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))
  # top_k_range needs to start from 1 to top_k_int
  top_k_range = tf.range(top_k_int) + 1
  top_k_range = tf.reshape(top_k_range, [-1, 1])
  # cast top_k_range to float
  top_k_range = tf.cast(top_k_range, dtype=tf.float32)
  cg_discount = tf.log(top_k_range + 1.0) / log_2
  return cg_discount


def _get_relevance_scores(scores):
  gain = 2 ** scores - 1
  return tf.reshape(gain, [-1, 1])


def safe_log(raw_scores, name=None):
  """
  Calculate log of a tensor, handling cases that
  raw_scores are close to 0s
  Args:
    raw_scores: An float `Tensor`.
  Returns:
    A float `Tensor` that hols the safe log base e of input
  """
  epsilon = 1E-8
  # clipped_raw_scores = raw_scores + epsilon
  clipped_raw_scores = tf.maximum(raw_scores, epsilon)
  return tf.log(clipped_raw_scores)
