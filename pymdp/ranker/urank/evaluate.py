"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger, load_best_ndcgs
from model.evaluation import evaluate
from model.reader import input_fn
from model.reader import load_dataset_from_tfrecords
from model.modeling import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--residual_model_dir', default='experiments/residual_model',
                    help="Directory containing params.json")
# loss functions
# grank, urrank, ranknet, listnet, listmle, lambdarank, mdprank
parser.add_argument('--loss_fn', default='grank',
                    help="model loss function")
# tf data folder for
# OHSUMED, MQ2007, MSLR-WEB10K, MSLR-WEB30K
parser.add_argument('--data_dir', default='../data/OHSUMED/5',
                    help="Directory containing the dataset")
# OHSUMED, MQ2007, MSLR-WEB10K, MSLR-WEB30K
parser.add_argument('--tfrecords_filename', default='OHSUMED.tfrecords',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of the best weights")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    if params.mlp_sizes is None or len(params.mlp_sizes) == 0:
        logging.error('mlp_sizes are not set correctly, at least one MLP layer is required')
    params.dict['loss_fn'] = args.loss_fn
    if params.num_learners > 1:
        params.dict['use_residual'] = True
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
    eval_inputs = input_fn('test', eval_dataset, params)
    logging.info("- done.")
    # Define the model
    logging.info("Creating the model...")
    weak_learner_id = load_best_ndcgs(os.path.join(args.model_dir, args.restore_from, 'learner.json'))[0]
    eval_model_spec = model_fn('test', eval_inputs, params, reuse=False, \
        weak_learner_id=int(weak_learner_id))
    # node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # print(node_names)
    logging.info("- done.")
    logging.info("Starting evaluation")
    logging.info("Optimized using {} learners".format(weak_learner_id))
    evaluate(eval_model_spec, args.model_dir, params, args.restore_from)
