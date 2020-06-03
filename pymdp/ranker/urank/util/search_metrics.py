
"""
Module containing tensorflow ranking metrics.
This module conforms to conventions used by tf.metrics.*.
In particular, each metric constructs two subgraphs: value_op and update_op:
  - The value op is used to fetch the current metric value.
  - The update_op is used to accumulate into the metric.

Note: similar to tf.metrics.*, metrics in here do not support multi-label learning.
We will have to write wrapper classes to create one metric per label.

Note: similar to tf.metrics.*, batches added into a metric via its update_op are cumulative!

"""
from __future__ import absolute_import, division

from collections import OrderedDict
from functools import partial

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

from util import math_fns


def ndcg(labels, predictions,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None,
                  top_ks=[1, 3, 5, 10],
                  use_predicted_order=False):
  # pylint: disable=unused-argument
  """
  Compute full normalized discounted cumulative gain (ndcg) based on predictions
    ndcg = dcg_k/idcg_k, k is a cut off ranking postion
    There are a few variants of ndcg
    The dcg (discounted cumulative gain) formula used in 
    twml.contrib.metrics.ndcg is
    \sum_{i=1}^k \frac{2^{relevance\_score} -1}{\log_{2}(i + 1)}
    k is the length of items to be ranked in a batch/query
    Notice that whether k will be replaced with a fixed value requires discussions 
    The scores in predictions are transformed to order and relevance scores to calculate ndcg
    A relevance score means how relevant a DataRecord is to a particular query       
  Args:
    labels: the ground truth value.
    predictions: the predicted values, whose shape must match labels. Ignored for CTR computation.
    metrics_collections: optional list of collections to add this metric into.
    updates_collections: optional list of collections to add the associated update_op into.
    name: an optional variable_scope name.

  Return:
    ndcg: A `Tensor` representing the ndcg score.
    update_op: A update operation used to accumulate data into this metric.
  """
  with tf.variable_scope(name, 'ndcgs', (labels, predictions)):
    label_scores = tf.to_float(labels, name='labels_to_float')
    predicted_scores = tf.to_float(predictions, name='predictions_to_float')

    total_ndcgs = _metric_variable([len(top_ks), 1], dtypes.float32, name='total_ndcgs')
    count_query = _metric_variable([], dtypes.float32, name='query_count')
    #-------------------------------------------------------------------------------->
    # # actual ndcg cutoff position top_k_int
    # max_prediction_size = array_ops.size(predicted_scores)
    # top_k_int = tf.minimum(max_prediction_size, top_k_int)
    # the ndcg score of the batch
    ndcgs = math_fns.cal_ndcg(label_scores,
      predicted_scores, top_ks=top_ks, use_predicted_order=use_predicted_order)
    #<--------------------------------------------------------------------------------
    # add ndcg of the current batch to total_ndcgs
    update_total_op = state_ops.assign_add(total_ndcgs, ndcgs)
    with ops.control_dependencies([ndcgs]):
      # count_query stores the number of queries
      # count_query increases by 1 for each batch/query
      update_count_op = state_ops.assign_add(count_query, 1)

    mean_ndcgs = math_fns.safe_div(total_ndcgs, count_query, 'mean_ndcgs')
    update_op = math_fns.safe_div(update_total_op, update_count_op, 'update_mean_ndcg_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, mean_ndcgs)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_ndcgs, update_op


def dcg(labels, predictions,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None):
  # pylint: disable=unused-argument
  """
  Compute full normalized discounted cumulative gain (ndcg) based on predictions
    ndcg = dcg_k/idcg_k, k is a cut off ranking postion
    There are a few variants of ndcg
    The dcg (discounted cumulative gain) formula used in 
    twml.contrib.metrics.ndcg is
    \sum_{i=1}^k \frac{2^{relevance\_score} -1}{\log_{2}(i + 1)}
    k is the length of items to be ranked in a batch/query
    Notice that whether k will be replaced with a fixed value requires discussions 
    The scores in predictions are transformed to order and relevance scores to calculate ndcg
    A relevance score means how relevant a DataRecord is to a particular query       
  Args:
    labels: the ground truth value.
    predictions: the predicted values, whose shape must match labels. Ignored for CTR computation.
    metrics_collections: optional list of collections to add this metric into.
    updates_collections: optional list of collections to add the associated update_op into.
    name: an optional variable_scope name.

  Return:
    dcg: A `Tensor` representing the dcg score.
    update_op: A update operation used to accumulate data into this metric.
  """
  with tf.variable_scope(name, 'dcg', (labels, predictions)):
    label_scores = tf.to_float(labels, name='label_to_float_dcg')
    predicted_scores = tf.to_float(predictions, name='predictions_to_float_dcg')

    total_dcg = _metric_variable([], dtypes.float32, name='total_dcg')
    count_query = _metric_variable([], dtypes.float32, name='query_count_dcg')

    # actual dcg cutoff position top_k_int is equal to max_prediction_size
    max_prediction_size = array_ops.size(predicted_scores)
    top_k_int = max_prediction_size
    # the dcg score of the batch
    dcg_full = math_fns.cal_dcg(label_scores,
      predicted_scores, top_k_int=top_k_int)
    # add dcg of the current batch to total_dcg
    update_total_op = state_ops.assign_add(total_dcg, dcg_full)
    with ops.control_dependencies([dcg_full]):
      # count_query stores the number of queries
      # count_query increases by 1 for each batch/query
      update_count_op = state_ops.assign_add(count_query, 1)

    mean_ndcg = math_fns.safe_div(total_dcg, count_query, 'mean_dcg')
    update_op = math_fns.safe_div(update_total_op, update_count_op, 'update_mean_dcg_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, mean_ndcg)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_ndcg, update_op


def idcg(labels, predictions,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None):
  # pylint: disable=unused-argument
  """
  Compute full normalized discounted cumulative gain (ndcg) based on predictions
    ndcg = dcg_k/idcg_k, k is a cut off ranking postion
    There are a few variants of ndcg
    The dcg (discounted cumulative gain) formula used in 
    twml.contrib.metrics.ndcg is
    \sum_{i=1}^k \frac{2^{relevance\_score} -1}{\log_{2}(i + 1)}
    k is the length of items to be ranked in a batch/query
    Notice that whether k will be replaced with a fixed value requires discussions 
    The scores in predictions are transformed to order and relevance scores to calculate ndcg
    A relevance score means how relevant a DataRecord is to a particular query       
  Args:
    labels: the ground truth value.
    predictions: the predicted values, whose shape must match labels. Ignored for CTR computation.
    metrics_collections: optional list of collections to add this metric into.
    updates_collections: optional list of collections to add the associated update_op into.
    name: an optional variable_scope name.

  Return:
    idcg: A `Tensor` representing the idcg score.
    update_op: A update operation used to accumulate data into this metric.
  """
  with tf.variable_scope(name, 'idcg', (labels, predictions)):
    label_scores = tf.to_float(labels, name='label_to_float_idcg')
    predicted_scores = tf.to_float(predictions, name='predictions_to_float_idcg')

    total_idcg = _metric_variable([], dtypes.float32, name='total_idcg')
    count_query = _metric_variable([], dtypes.float32, name='query_count_idcg')

    # actual idcg cutoff position top_k_int is equal to max_prediction_size
    max_prediction_size = array_ops.size(predicted_scores)
    top_k_int = max_prediction_size
    # the idcg score of the batch
    idcg_full = math_fns.cal_idcg(label_scores,
      predicted_scores, top_k_int=top_k_int)
    # add idcg of the current batch to total_idcg
    update_total_op = state_ops.assign_add(total_idcg, idcg_full)
    with ops.control_dependencies([idcg_full]):
      # count_query stores the number of queries
      # count_query increases by 1 for each batch/query
      update_count_op = state_ops.assign_add(count_query, 1)

    mean_ndcg = math_fns.safe_div(total_idcg, count_query, 'mean_idcg')
    update_op = math_fns.safe_div(update_total_op, update_count_op, 'update_mean_idcg_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, mean_ndcg)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_ndcg, update_op


def err(labels, predictions,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None,
                  top_k_int=1,
                  use_predicted_order=False):
  # pylint: disable=unused-argument
  """
  Compute Expected Reciprocal Rank
    The scores in predictions are transformed to order and relevance scores to calculate ndcg
    A relevance score means how relevant a DataRecord is to a particular query       
  Args:
    labels: the ground truth value.
    predictions: the predicted values, whose shape must match labels. Ignored for CTR computation.
    metrics_collections: optional list of collections to add this metric into.
    updates_collections: optional list of collections to add the associated update_op into.
    name: an optional variable_scope name.

  Return:
    err: A `Tensor` representing the err score.
    update_op: A update operation used to accumulate data into this metric.
  """
  with tf.variable_scope(name, 'err', (labels, predictions)):
    label_scores = tf.to_float(labels, name='label_to_float_err')
    predicted_scores = tf.to_float(predictions, name='predictions_to_float_err')

    total_err = _metric_variable([], dtypes.float32, name='total_err')
    count_query = _metric_variable([], dtypes.float32, name='query_count_err')

    # actual err cutoff position top_k_int is equal to max_prediction_size
    max_prediction_size = array_ops.size(predicted_scores)
    top_k_int = tf.minimum(max_prediction_size, top_k_int)
    # the err score of the batch

    err_full = math_fns.cal_err(labels,
      predicted_scores, top_k_int=top_k_int, use_predicted_order=use_predicted_order)

    # add err of the current batch to total_err
    update_total_op = state_ops.assign_add(total_err, err_full)
    with ops.control_dependencies([err_full]):
      # count_query stores the number of queries
      # count_query increases by 1 for each batch/query
      update_count_op = state_ops.assign_add(count_query, 1)

    mean_err = math_fns.safe_div(total_err, count_query, 'mean_err')
    update_op = math_fns.safe_div(update_total_op, update_count_op, 'update_mean_err_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, mean_err)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_err, update_op

# Copied from metrics_impl.py with minor modifications.
# https://github.com/tensorflow/tensorflow/blob/v1.5.0/tensorflow/python/ops/metrics_impl.py#L39
def _metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""

  return tf.Variable(
    lambda: tf.zeros(shape, dtype),
    trainable=False,
    collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
    validate_shape=validate_shape,
    name=name)


# binary metric_name: (metric, requires thresholded output)
SUPPORTED_BINARY_CLASS_METRICS = {
  # thresholded metrics
  'accuracy': (tf.metrics.accuracy, True),
  'precision': (tf.metrics.precision, True),
  'recall': (tf.metrics.recall, True),
}

# search metric_name: metric
SUPPORTED_SEARCH_METRICS = {
  # ndcg needs the raw prediction scores to sort
  'ndcg': ndcg,
  'err': err,
  # 'dcg': dcg,
  # 'idcg': idcg,
}


def get_search_metric_fn(labels, predictions,
  binary_metrics=None, search_metrics=None,
  ndcg_top_ks=[1, 3, 5, 10], use_ndcg_metrics=True,
  use_binary_metrics=False, use_err_metric=True, use_predicted_order=False):
  """
  Returns a function having signature:

  .. code-block:: python

    def get_eval_metric_ops(graph_output, labels, weights):
      ...
      return eval_metric_ops

  Args:
    only used in pointwise learning-to-rank
    binary_metrics (list of String):
      a list of metrics of interest. E.g. ['accuracy']
    search_metrics (list of String):
      a list of metrics of interest. E.g. ['ndcg']
      These metrics are evaluated and reported to tensorboard *during the eval phases only*.
      Supported metrics:
        - ndcg
      NOTE: ndcg works for ranking-relatd problems.
      A batch contains all DataRecords that belong to the same query
    ndcg_top_ks (list of integers):
      The cut-off ranking postions for a query
      When ndcg_top_ks is None or empty (the default), it defaults to [1, 3, 5, 10]
    use_binary_metrics:
      False (default)
      Only set it to true in pointwise learning-to-rank
  """
  # pylint: disable=dict-keys-not-iterating

  if ndcg_top_ks is None or not ndcg_top_ks:
    ndcg_top_ks = [1, 3, 5, 10]

  if search_metrics is None:
    search_metrics = list(SUPPORTED_SEARCH_METRICS.keys())

  if binary_metrics is None and use_binary_metrics:
    # Added SUPPORTED_BINARY_CLASS_METRICS in twml.metics as well
    # they are only used in pointwise learing-to-rank
    binary_metrics = list(SUPPORTED_BINARY_CLASS_METRICS.keys())

    """
    graph_output:
      dict that is returned by build_graph given input features.
    labels:
      target labels associated to batch.
    weights:
      weights of the samples..
    """

  eval_metric_ops = OrderedDict()

  threshold = 0.5

  # hard_preds is a tensor
  hard_preds = tf.greater_equal(predictions, threshold)

  # add search metrics to eval_metric_ops dict
  for metric_name in search_metrics:
    metric_name = metric_name.lower()  # metric name are case insensitive.

    if metric_name in eval_metric_ops:
      # avoid adding duplicate metrics.
      continue

    search_metric_factory = SUPPORTED_SEARCH_METRICS.get(metric_name)
    if search_metric_factory:
      if metric_name == 'ndcg':
        if use_ndcg_metrics == True:
          # for top_k in ndcg_top_ks:
          metric_name_ndcg_top_k = metric_name
          # metric name will show as ndcg_1, ndcg_10, ...
          # metric_name_ndcg_top_k = metric_name + '_' + str(top_k)
          # top_k_int = tf.constant(top_k, dtype=tf.int32)
          # # Note: having weights in ndcg does not make much sense
          # # Because ndcg already has position weights/discounts
          # # Thus weights are not applied in ndcg metric
          value_op, update_op = search_metric_factory(
            labels=labels,
            predictions=predictions,
            name=metric_name_ndcg_top_k,
            top_ks=ndcg_top_ks,
            use_predicted_order=use_predicted_order)
          eval_metric_ops[metric_name_ndcg_top_k] = (value_op, update_op)
      elif metric_name == 'err':
        if use_err_metric == True:
          for top_k in ndcg_top_ks:
            # metric name will show as err_1, err_10, ...
            metric_name_err_top_k = metric_name + '_' + str(top_k)
            top_k_int = tf.constant(top_k, dtype=tf.int32)
            # # Note: having weights in err does not make much sense
            # # Because err already has position weights/discounts
            # # Thus weights are not applied in err metric
            value_op, update_op = search_metric_factory(
              labels=labels,
              predictions=predictions,
              name=metric_name_err_top_k,
              top_k_int=top_k_int,
              use_predicted_order=use_predicted_order)
            eval_metric_ops[metric_name_err_top_k] = (value_op, update_op)       
      else:
        metric_name = metric_name.lower()
        value_op, update_op = search_metric_factory(
          labels=labels,
          predictions=predictions,
          name=metric_name)
        eval_metric_ops[metric_name] = (value_op, update_op)          
    else:
      raise ValueError('Cannot find the search metric named ' + metric_name)

  if use_binary_metrics:
    # add binary metrics to eval_metric_ops dict
    for metric_name in binary_metrics:

      if metric_name in eval_metric_ops:
        # avoid adding duplicate metrics.
        continue

      metric_name = metric_name.lower()  # metric name are case insensitive.
      binary_metric_factory, requires_threshold = SUPPORTED_BINARY_CLASS_METRICS.get(metric_name)
      if binary_metric_factory:
        value_op, update_op = binary_metric_factory(
          labels=labels,
          predictions=(hard_preds if requires_threshold else predictions),
          name=metric_name)
        eval_metric_ops[metric_name] = (value_op, update_op)
      else:
        raise ValueError('Cannot find the binary metric named ' + metric_name)

  return eval_metric_ops
