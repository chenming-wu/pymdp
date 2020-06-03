# python model/reader.py
import os
import argparse
import numpy as np
import tensorflow as tf
import argparse
import logging

from model.utils import Params
# from utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


def _parse_function(serialized_example_proto):
    keys_to_features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'feature_raw': tf.FixedLenFeature([], tf.string),
        'label_gain': tf.VarLenFeature(tf.float32),
        'unique_rating': tf.FixedLenFeature([], tf.int64),
        'label': tf.VarLenFeature(tf.float32)
        }
    inputs = tf.parse_single_example(
      serialized_example_proto,
      # Defaults are not specified since both keys are required.
      features=keys_to_features)
    height = tf.cast(inputs['height'], tf.int32)
    # height = tf.reshape(height, [])
    width = tf.cast(inputs['width'], tf.int32)
    # width = tf.reshape(width, [])
    feature = tf.decode_raw(inputs['feature_raw'], tf.float32)
    # feature = tf.reshape(feature, [height, width])
    label_sparse = tf.cast(inputs['label'], tf.float32)
    # label = tf.sparse.to_dense(label_sparse, validate_indices=False)
    label = tf.sparse_tensor_to_dense(label_sparse, validate_indices=False)
    label_gain_sparse = tf.cast(inputs['label_gain'], tf.float32)
    # label_gain = tf.sparse.to_dense(label_gain_sparse, validate_indices=False)
    label_gain = tf.sparse_tensor_to_dense(label_gain_sparse, validate_indices=False)
    unique_rating = tf.cast(inputs['unique_rating'], tf.int32)
    # no need to add padding later
    return feature, label, height, width, label_gain, unique_rating



def load_dataset_from_tfrecords(path_tfrecords_filename):
    # tfrecords_filename
    # file_type + "_" + tfrecords_filename
    dataset = tf.data.TFRecordDataset(path_tfrecords_filename)   
    # Parse the record into tensors.
    dataset = dataset.map(_parse_function)
    return dataset 


def input_fn(mode, dataset, params):
    # Shuffle the dataset
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1
    if mode != 'test':
        dataset = dataset.shuffle(buffer_size=buffer_size)
    # batch_size = 1 
    # Generate batches
    dataset = dataset.batch(params.batch_size)
    # Repeat the input ## num_epochs times
    dataset = dataset.repeat()
    # prefetch a batch
    dataset = dataset.prefetch(1)
    # # Create a one-shot iterator
    # iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    # Get batch X and Y
    features, labels, height, width, label_gains, unique_rating = iterator.get_next()
    width = tf.squeeze(width)
    height = tf.squeeze(height)
    features = tf.reshape(features, [height, width])
    labels = tf.reshape(labels, [-1, 1])
    label_gains = tf.reshape(label_gains, [-1, 1])
    unique_rating = tf.squeeze(unique_rating)

    iterator_init_op = iterator.initializer
    inputs = {
        'features': features,
        'labels': labels,
        'height': height,
        'width': width,
        'unique_rating': unique_rating,
        'label_gains': label_gains,
        # 'use_predicted_order':False,
        'iterator_init_op': iterator_init_op
    }
    return inputs   

def _shuffle_docs(labels, features, height, width):
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

if __name__ == "__main__":
    tf.set_random_seed(230)
    dataset = load_dataset_from_tfrecords("../data/OHSUMED/0/test_OHSUMED.tfrecords")
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    mode = 'eval'
    # iterator_init_op = iterator.initializer    
    inputs = input_fn(mode, dataset, params)
    iterator_init_op = inputs['iterator_init_op']
    features, labels, label_gains, unique_rating = inputs['features'], inputs['labels'], inputs['label_gains'],  inputs['unique_rating']
    logging.info("- done loading dataset.")
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(iterator_init_op)
        for i in range(3):  
            print(sess.run([label_gains, labels, unique_rating]))
