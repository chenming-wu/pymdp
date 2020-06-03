"""Evaluate the model"""

import argparse
import logging
import os
import warnings
os.environ['KMP_WARNINGS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']="-1"

from prepare_data import normalize_min_max_feature_array, normalize_mean_max_feature_array

warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

from model.utils import Params
from model.utils import set_logger, load_best_ndcgs
from model.reader import input_fn
from model.reader import load_dataset_from_tfrecords
from model.modeling import model_fn

import logging
import os

from model.utils import save_dict_to_json, save_predictions_to_file
from model.utils import get_expaned_metrics

class EvaluatePointConfig:
    def __init__(self):
        self.model_dir = "E:/RAL2020/ranker/src/experiments/base_model"
        self.residual_model_dir = 'E:/RAL2020/ranker/src/experiments/residual_model'
        self.loss_fn = "urank"
        self.data_dir = "E:/RAL2020/ranker/data/RAL-6/1"
        self.tfrecords_filename = 'RAL.tfrecords'
        self.restore_from = 'best_weights'

"""Tensorflow utility functions for evaluation"""


def evaluate_sess(sess, model_spec, num_steps, features, labels, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()
    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    # sess.run([model_spec['iterator_init_op'], model_spec['metrics_init_op']])

    # sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    if params.save_predictions:
        # save the predictions and lable_qid to files
        prediction_list = []
        label_list = []
        # compute metrics over the dataset
        for temp_query_id in range(int(1)):
            # prediction_per_query, label_per_query, height = sess.run([predictions, labels, model_spec["height"]])
            # logging.info("- height per query: \n" + str(height))
            prediction_per_query, _ = sess.run([model_spec["predictions"],
                                                                              update_metrics], feed_dict={
                                                                                "features:0": features,
                                                                                "labels:0": labels,
                                                                                "height:0": features.shape[0],
                                                                                "width:0": features.shape[1],
                                                                                "unique_rating:0": len(set(labels)),
                                                                                "label_gains:0": [2**v-1 for v in labels]})
            return prediction_per_query
        save_predictions_to_file(prediction_list, "./prediction_output")
        # tensorflow mess up test input orders
        save_predictions_to_file(label_list, "./label_output")
    else:
        # only update metrics
        for temp_query_id in range(int(num_steps)):
            sess.run(update_metrics)
    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    expanded_metrics_val = get_expaned_metrics(metrics_val, params.top_ks)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in expanded_metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)
    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in expanded_metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)
    return expanded_metrics_val


class EvalPoint:
    def __init__(self):
        # Load the parameters
        args = EvaluatePointConfig()
        json_path = os.path.join(args.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)
        if params.mlp_sizes is None or len(params.mlp_sizes) == 0:
            logging.error('mlp_sizes are not set correctly, at least one MLP layer is required')
        params.dict['loss_fn'] = args.loss_fn

        # Load the parameters from the dataset, that gives the size etc. into params
        json_path = os.path.join(args.data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
        params.update(json_path)
        # Set the logger
        set_logger(os.path.join(args.model_dir, 'evaluate.log'))
        # # Get paths for tfrecords
        path_eval_tfrecords = os.path.join(args.data_dir, 'test_' + args.tfrecords_filename)
        # Create the input data pipeline
        logging.info("Creating the dataset...")
        eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
        # Create iterator over the test set
        # eval_inputs = input_fn('test', eval_dataset, params)
        eval_inputs = online_input_fn()
        logging.info("- done.")
        # print(type(eval_inputs))

        # Define the model
        logging.info("Creating the model...")
        weak_learner_id = load_best_ndcgs(os.path.join(args.model_dir, args.restore_from, 'learner.json'))[0]
        self.model_spec = model_fn('test', eval_inputs, params, reuse=False, weak_learner_id=int(weak_learner_id))
        # node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # print(node_names)
        logging.info("- done.")
        logging.info("Starting evaluation")
        logging.info("Optimized using {} learners".format(weak_learner_id))
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.params = params
        self.sess.run(self.model_spec['variable_init_op'])
        save_path = os.path.join(args.model_dir, args.restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        self.saver.restore(self.sess, save_path)

    def evaluate(self, features):
        num_steps = 1
        features = normalize_min_max_feature_array(features)
        n_features = features.shape[0]
        labels = [0 for i in range(n_features)]
        predicted_scores = evaluate_sess(self.sess, self.model_spec, num_steps, features, labels, params=self.params)
        predicted_scores = np.squeeze(predicted_scores)
        return np.argsort(predicted_scores)[::-1]

def online_input_fn():
    features = tf.placeholder(tf.float32, name="features")
    labels = tf.placeholder(tf.float32, name="labels")
    height = tf.placeholder(tf.int32, name="height")
    width = tf.placeholder(tf.int32, name="width")
    unique_rating = tf.placeholder(tf.int32, name="unique_rating")
    label_gains = tf.placeholder(tf.float32, name="label_gains")
    inputs = {
        'features': features,
        'labels': labels,
        'height': height,
        'width': width,
        'unique_rating': unique_rating,
        'label_gains': label_gains,
    }
    return inputs


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)
    evaluator = EvalPoint()
    features = np.array([[0, 0, 0, 0, 0, 0, 0.238741, 0.270131, 0.346482, 0.025711, 0.000000, 0.270131],
                         [0, 0, 0, 0, 0, 0, 0.242391, 0.298806, 0.327038, 0.098476, 0.000000, 0.298806]])

    res = evaluator.evaluate(features)
    print("evaluated result = ", res)
