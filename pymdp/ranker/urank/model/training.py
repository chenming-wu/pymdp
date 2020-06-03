"""Tensorflow utility functions for training"""
# tensorboard --logdir=experiments/base_model/
# tensorboard --logdir=experiments/base_model/train_summaries
# tensorboard --logdir=experiments/base_model/eval_summaries

import logging
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json, load_best_ndcgs, get_expaned_metrics
from model.evaluation import evaluate_sess


def train_sess(sess, model_spec, num_steps, writer, params):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training

    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    # sess.run([model_spec['iterator_init_op'], model_spec['metrics_init_op']])
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])
    # Use tqdm for progress bar
    t = trange(int(num_steps))
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i == params.save_summary_steps - 1:
        # if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step])
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    expanded_metrics_val = get_expaned_metrics(metrics_val, params.top_ks)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in expanded_metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)


# NDCG@10
# def isSavingWeights(eval_metrics, best_eval_metrics):
#     if eval_metrics[len(eval_metrics) - 1] >= best_eval_metrics[len(eval_metrics) - 1]:
#         return True
#     return False

# NDCG@1, 3, 5, 10
def isSavingWeights(eval_metrics, best_eval_metrics):
    for i in range(len(eval_metrics)):
        if eval_metrics[i] > best_eval_metrics[i]:
            return True
        elif eval_metrics[i] < best_eval_metrics[i]:
            return False
        else:
            continue
    return False

def train_and_evaluate(train_model_spec, eval_model_spec,
    model_dir, params, learner_id=0, restore_from=None, global_epoch=1):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])
        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'vali_summaries'), sess.graph)
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")

        best_eval_metric = 0.0 # ndcg_1
        # best_loss_metric = float('inf')
        second_eval_metric = 0.0 # ndcg_3
        third_eval_metric = 0.0 # ndcg_5
        forth_eval_metric = 0.0 # ndcg_10
        # global_epoch = 0
        # Reload weights from directory if specified
        # restor from the previous learner
        if restore_from is not None:
            save_path = os.path.join(model_dir, restore_from)
            if os.path.isdir(save_path):
                save_path = tf.train.latest_checkpoint(save_path)
                # begin_at_epoch = int(restore_from.split('-')[-1])           
            logging.info("Restoring parameters from {}".format(save_path))
            # last_saver = tf.train.import_meta_graph(save_path+".meta")
            pretrained_include = ['model/mlp']
            if params.loss_fn=='urrank':
                pretrained_include = ['model/ur']
            for i in range(1, learner_id):
                pretrained_include.append('residual_mlp_{}'.format(learner_id))

            pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=pretrained_include)
            pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
            pretrained_saver.restore(sess, save_path)
            [best_eval_metric, second_eval_metric, third_eval_metric, forth_eval_metric] = \
            load_best_ndcgs(best_json_path)
            # print('[best_eval_metric, second_eval_metric, third_eval_metric, forth_eval_metric]', \
            # [best_eval_metric, second_eval_metric, third_eval_metric, forth_eval_metric])
        # for each learner
        early_stopping_count = 0
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            if early_stopping_count == int(params.early_stoping_epochs):
                logging.info("Early stopping at learner {}, epoch {}/{}".format(learner_id, epoch + 1, \
                    begin_at_epoch + params.num_epochs))
                break
            # Run one epoch
            logging.info("Learner {}, Epoch {}/{}".format(learner_id, epoch + 1, \
                begin_at_epoch + params.num_epochs))
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_spec, num_steps, train_writer, params)
            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
            last_saver.save(sess, last_save_path, global_step=global_epoch)
            # Evaluate for one epoch on validation set
            num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
            metrics = evaluate_sess(sess, eval_model_spec, num_steps, eval_writer, params)
            # If best_eval, best_save_path
            # eval_metric = metrics['dcg']
            # loss_metric = metrics['loss']       
            eval_metric = round(metrics['ndcg_1'], 6)
            eval_metric_2 = round(metrics['ndcg_3'], 6)
            eval_metric_3 = round(metrics['ndcg_5'], 6)
            eval_metric_4 = round(metrics['ndcg_10'], 6)
            # eval_metric = metrics['ndcg_1']
            # eval_metric_2 = metrics['ndcg_3']
            # eval_metric_3 = metrics['ndcg_5']
            # eval_metric_4 = metrics['ndcg_10']    
            eval_metrics = [eval_metric, eval_metric_2, eval_metric_3, eval_metric_4]
            best_eval_metrics = [best_eval_metric, second_eval_metric, third_eval_metric, \
            forth_eval_metric]
            if isSavingWeights(eval_metrics, best_eval_metrics):
                # rest early_stopping_count
                early_stopping_count = 0
                # Store new best ndcg_1
                # this worsk better than eval_metric > best_eval_metric
                # and isSavingWeights
                best_eval_metric = eval_metric
                # loss_metric = best_loss_metric
                second_eval_metric = eval_metric_2
                third_eval_metric = eval_metric_3
                forth_eval_metric = eval_metric_4
                # Save weights
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
                best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                logging.info("- Found new best metric score, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics, best_json_path)
                save_dict_to_json({'stopped_at_learner': learner_id}, \
                    os.path.join(model_dir, 'best_weights', 'learner.json'))
            else:
                early_stopping_count = early_stopping_count + 1
            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
            global_epoch += 1
    return global_epoch
