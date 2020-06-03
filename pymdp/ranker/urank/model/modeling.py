"""Define the model."""
import sys, random, logging
import tensorflow as tf

from util import loss_fns, math_fns, scores, sample, search_metrics, masks
from tensorflow.python.ops import array_ops

def build_residual_model(is_training, inputs, params, weak_learner_id):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, residuals...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! boosting is only supported for grank and urrank
    """
    mse_loss = tf.constant(0.0, dtype=tf.float32)
    # MLP netowork for residuals
    features = inputs['features']
    features = tf.reshape(features, [-1, int(params.feature_dim)])
    if params.loss_fn == 'grank':
        predicted_scores = _get_mlp_logits(features, params)
    elif params.loss_fn == 'urrank':
        predicted_scores = _get_ur_logits(features, params)
    else:
        logging.error('Loss function not supported for boosting')
        sys.exit(1)
    if weak_learner_id >= 1:
        for trained_learner_id in range(1, weak_learner_id):
            predicted_scores += _get_residual_mlp_logits(features, params, \
            weak_learner_id=trained_learner_id)
        predicted_scores = tf.stop_gradient(predicted_scores)
        residual_predicted_scores = _get_residual_mlp_logits(features, params, \
            weak_learner_id=weak_learner_id)
        # boosted_scores = predicted_scores + 1/math.sqrt(weak_learner_id) * residual_predicted_scores
        boosted_scores = predicted_scores + residual_predicted_scores
    else:
        boosted_scores = predicted_scores
    if not is_training:
        return boosted_scores, mse_loss
    if weak_learner_id >= 1:
        labels = inputs['labels']
        residuals = get_residual(labels, predicted_scores)
        mse_loss = tf.losses.mean_squared_error(residuals, residual_predicted_scores)
    return boosted_scores, mse_loss


def build_g_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! when using the build_model mdprank needs a learning_rate around 1e-5 - 1e-7
    """
    permutation_loss = tf.constant(0, dtype=tf.float32)
    # MLP netowork
    features = inputs['features']
    features = tf.reshape(features, [-1, int(params.feature_dim)])
    predicted_scores = _get_mlp_logits(features, params)
    if not is_training:
        return predicted_scores, permutation_loss
    labels = inputs['labels']
    permutation_loss = get_permutation_loss(labels, predicted_scores)
    return predicted_scores, permutation_loss

def _get_residual_mlp_logits(features, params, weak_learner_id=1):
    # features = tf.reshape(features, [-1, params.feature_dim])
    with tf.variable_scope('residual_mlp_{}'.format(weak_learner_id), reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(features, params.residual_mlp_sizes[0],
            name='residual_{}_dense_0'.format(weak_learner_id), activation=tf.nn.relu)
        for i in range(1, len(params.residual_mlp_sizes)):
            out = tf.layers.dense(out, params.residual_mlp_sizes[i], \
                name='residual_{}_dense_{}'.format(weak_learner_id, i), activation=tf.nn.relu)
        logits = tf.layers.dense(out, 1,
            name='residual_{}_dense_{}'.format(weak_learner_id, len(params.residual_mlp_sizes)))
    return logits

def _get_mlp_logits(features, params):
    with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(features, params.mlp_sizes[0], \
            name='dense_0', activation=tf.nn.relu)
        for i in range(1, len(params.mlp_sizes)):
            out = tf.layers.dense(out, params.mlp_sizes[i], \
                name='dense_{}'.format(i), activation=tf.nn.relu)
        logits = tf.layers.dense(out, 1, \
            name='dense_{}'.format(len(params.mlp_sizes)))
    return logits

def get_permutation_loss(labels, predicted_scores):
    tmp_labels = tf.cast(tf.squeeze(labels), tf.int32)
    y, idx = tf.unique(tmp_labels)
    which_rank = tf.shape(y)[0] - 1
    which_rank = tf.cast(which_rank, tf.float32)  
    raw_pairwise_label_scores = scores.get_pairwise_scores(labels)
    mask = masks.list_mask(raw_pairwise_label_scores)
    tmp_denominator = tf.matmul(mask, tf.exp(predicted_scores))
    multi_label_loss = tf.log(1 + tmp_denominator / tf.exp(predicted_scores))
    multi_label_loss *= 2**labels - 1
    permutation_loss = tf.reduce_sum(multi_label_loss)
    permutation_loss /= which_rank
    return permutation_loss

def get_lambda_permutation_loss(labels, predicted_scores):
    tmp_labels = tf.cast(tf.squeeze(labels), tf.int32)
    y, idx = tf.unique(tmp_labels)
    which_rank = tf.shape(y)[0] - 1
    which_rank = tf.cast(which_rank, tf.float32)
    raw_pairwise_label_scores = scores.get_pairwise_scores(labels)
    mask = masks.list_mask(raw_pairwise_label_scores)
    # calculate delta_z
    gains = 2**labels - 1
    raw_pairwise_scores = scores.get_pairwise_scores(predicted_scores)
    score_mask = masks.list_mask(raw_pairwise_scores)
    ranks = tf.reduce_sum(score_mask, axis=1)
    ranks = tf.reshape(ranks, [-1, 1])
    log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))
    cg_discounts = tf.log(ranks + 2.0) / log_2
    cg_discounts = tf.reshape(cg_discounts, [-1, 1])
    raw_pairwise_discounts = tf.abs(scores.get_pairwise_scores(cg_discounts))
    raw_pairwise_gains = tf.abs(scores.get_pairwise_scores(gains))
    delta_z = raw_pairwise_gains
    delta_z = tf.multiply(raw_pairwise_discounts, raw_pairwise_gains)
    abs_pairwise_scores = tf.abs(raw_pairwise_scores)
    delta_z = tf.divide(delta_z, abs_pairwise_scores+1e-7)
    # max_dcg = tf.reduce_sum(math_fns.cal_idcg_ks(labels, 10))
    max_dcg = tf.reduce_sum(math_fns.cal_idcg_ks(labels, array_ops.size(predicted_scores)))
    delta_z = tf.divide(delta_z, max_dcg)
    delta_mask = tf.multiply(mask, delta_z)
    delta_mask = tf.stop_gradient(delta_mask)

    tmp_denominator = tf.matmul(delta_mask, tf.exp(predicted_scores))
    multi_label_loss = tf.log(1 + tmp_denominator / tf.exp(predicted_scores))
    permutation_loss = tf.reduce_sum(multi_label_loss)
    permutation_loss /= which_rank
    # # equal pairwise loss
    # pairwise_prediction_scores = scores.get_pairwise_scores(predicted_scores)
    # pairwise_equal_loss = loss_fns.get_equal_pair_loss(raw_pairwise_label_scores, \
    #     pairwise_prediction_scores)
    # permutation_loss += pairwise_equal_loss
    return permutation_loss

def get_residual(labels, predicted_scores):
    tmp_labels = tf.cast(tf.squeeze(labels), tf.int32)
    y, idx = tf.unique(tmp_labels)
    which_rank = tf.shape(y)[0] - 1
    which_rank = tf.cast(which_rank, tf.float32)
    exp_predicted_scores = tf.exp(predicted_scores)
    raw_pairwise_label_scores = scores.get_pairwise_scores(labels)
    # gradient with respect to d
    d_mask = masks.list_mask(raw_pairwise_label_scores)
    tmp_denominator = tf.matmul(d_mask, exp_predicted_scores)
    # loss_score_gradients
    residuals = 1 / (1 + exp_predicted_scores / tmp_denominator)
    gains = 2**labels - 1
    residuals *= gains
    # gradient with respect to d'
    _tmp_denominator = tmp_denominator + exp_predicted_scores
    _d_mask= masks.list_negative_mask(raw_pairwise_label_scores)
    _residuals = tf.matmul(_d_mask, gains / _tmp_denominator)
    _residuals *= exp_predicted_scores
    residuals += _residuals
    residuals /= which_rank
    return residuals

def get_lambda_residual(labels, predicted_scores):
    tmp_labels = tf.cast(tf.squeeze(labels), tf.int32)
    y, idx = tf.unique(tmp_labels)
    which_rank = tf.shape(y)[0] - 1
    which_rank = tf.cast(which_rank, tf.float32)
    exp_predicted_scores = tf.exp(predicted_scores)
    raw_pairwise_label_scores = scores.get_pairwise_scores(labels)
    # gradient with respect to d
    d_mask = masks.list_mask(raw_pairwise_label_scores)
    # calculate delta_z
    gains = 2**labels - 1
    raw_pairwise_scores = scores.get_pairwise_scores(predicted_scores)
    score_mask = masks.list_mask(raw_pairwise_scores)
    ranks = tf.reduce_sum(score_mask, axis=1)
    ranks = tf.reshape(ranks, [-1, 1])
    log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))
    cg_discounts = tf.log(ranks + 2.0) / log_2
    cg_discounts = tf.reshape(cg_discounts, [-1, 1])
    raw_pairwise_discounts = tf.abs(scores.get_pairwise_scores(cg_discounts))
    raw_pairwise_gains = tf.abs(scores.get_pairwise_scores(gains))
    delta_z = raw_pairwise_gains
    delta_z = tf.multiply(raw_pairwise_discounts, raw_pairwise_gains)
    abs_pairwise_scores = tf.abs(raw_pairwise_scores)
    delta_z = tf.divide(delta_z, abs_pairwise_scores+1e-7)
    # max_dcg = tf.reduce_sum(math_fns.cal_idcg_ks(labels, 10))
    max_dcg = tf.reduce_sum(math_fns.cal_idcg_ks(labels, array_ops.size(predicted_scores)))
    delta_z = tf.divide(delta_z, max_dcg)
    delta_d_mask = tf.multiply(d_mask, delta_z)
    delta_d_mask = tf.stop_gradient(delta_d_mask)
    tmp_denominator = tf.matmul(d_mask, exp_predicted_scores)
    # loss_score_gradients
    residuals = 1 / (1 + exp_predicted_scores / tmp_denominator)
    gains = 2**labels - 1
    # gain_sum = tf.reduce_sum(gains)
    # residuals /= gain_sum
    # residuals /= which_rank
    # residuals /= num_not_min
    # gradient with respect to d'
    _tmp_denominator = tmp_denominator + exp_predicted_scores
    _d_mask= masks.list_negative_mask(raw_pairwise_label_scores)

    delta__d_mask = tf.multiply(_d_mask, delta_z)
    delta__d_mask = tf.stop_gradient(delta__d_mask)

    _residuals = tf.matmul(delta__d_mask, 1 / _tmp_denominator)
    _residuals *= exp_predicted_scores
    residuals += _residuals
    residuals /= which_rank
    return residuals


def build_u_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! when using the build_model mdprank needs a learning_rate around 1e-5 - 1e-7
    """
    permutation_loss = tf.constant(0, dtype=tf.float32)
    # MLP netowork
    features = inputs['features']
    features = tf.reshape(features, [-1, int(params.feature_dim)])
    predicted_scores = _get_mlp_logits(features, params)
    if not is_training:
        return predicted_scores, permutation_loss
    labels = inputs['labels']
    # global minimum label
    mini_label = tf.reduce_min(labels)
    n_data = inputs['height']
    # n_data = tf.shape(labels)[0]
    # pass logits to predicted_scores
    multi_label_loss = _get_multi_label_loss(labels, predicted_scores - tf.reduce_min(predicted_scores))
    permutation_loss += multi_label_loss
    keep_labels, keep_predicted_scores = _get_updated_predictions_labels(labels, predicted_scores)
    # prepare for the loop
    unique_rating = tf.constant(1, dtype=tf.float32)
    shape_invariants = [unique_rating.get_shape(), tf.TensorShape([None, 1]), \
    tf.TensorShape([None, 1]), permutation_loss.get_shape()]
    # while loop
    def _cond(unique_rating, keep_predicted_scores, keep_labels, permutation_loss):
        # the general case
        return tf.greater(tf.reduce_max(keep_labels), mini_label)
    # geneate loop boday
    def _gen_loop_body():
        def loop_body(unique_rating, keep_predicted_scores, keep_labels, permutation_loss):
            multi_label_loss = _get_multi_label_loss(keep_labels, keep_predicted_scores - \
                                tf.reduce_min(keep_predicted_scores))
            permutation_loss += multi_label_loss
            keep_labels, keep_predicted_scores = _get_updated_predictions_labels(keep_labels, keep_predicted_scores)
            return tf.add(unique_rating, 1.0), keep_predicted_scores, keep_labels, permutation_loss
        return loop_body
    # loop result
    unique_rating, keep_predicted_scores, keep_labels, permutation_loss = tf.while_loop(_cond, _gen_loop_body(),
    [unique_rating, keep_predicted_scores, keep_labels, permutation_loss], shape_invariants=shape_invariants)
    num_tasks = tf.shape(keep_labels)[0]
    permutation_loss /= unique_rating
    # return prediction scores and permutation_loss
    return predicted_scores, permutation_loss

def build_ur_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! when using the build_model mdprank needs a learning_rate around 1e-5 - 1e-7
    """
    permutation_loss = tf.constant(0, dtype=tf.float32)
    features = inputs['features']
    features = tf.reshape(features, [-1, int(params.feature_dim)])
    with tf.variable_scope('ur', reuse=tf.AUTO_REUSE):
        # mlp for rnn
        out_l0 = tf.layers.dense(features, params.mlp_sizes[0], name='full_dense_l0', activation=tf.nn.relu)
        out_l1 = tf.layers.dense(out_l0, params.mlp_sizes[-1], name='full_dense_l1', activation=tf.nn.relu)
        # dropout or batch normalization did not help
        # out_l1 = tf.nn.dropout(out_l1, params.dropout_rate)
        x = tf.squeeze(out_l1)
        x = tf.reshape(x, [-1, params.mlp_sizes[-1]])
        h_1 = tf.zeros(tf.shape(x))
        h_1 = tf.cast(h_1, dtype=tf.float32)
        # ls, current = gru(x, h_1, params)
        if params.rnn == 'C2':
            ls, current = rnnC2(x, h_1, params)
        else:
            ls, current = rnnC1(x, h_1, params)
        # use ls at the 1st step directly for validation and inference
        if not is_training:
            # since we use NDCG as metric for the best weights
            # we do not have to calculate permutation_loss
            # permutation_loss is 0 in validation and inference
            return ls, permutation_loss
        # the following is only for rating
        # calculate loss
        labels = inputs['labels']
        n_data = inputs['height']
        # n_data = tf.shape(labels)[0]       
        # global minimum label
        mini_label = tf.reduce_min(labels)
        total_predicted_scores = ls - tf.reduce_min(ls)
        # total_predicted_scores = ls
        inputs['no_ndcg'] = 1
        multi_label_loss = _get_multi_label_loss(labels, total_predicted_scores)
        permutation_loss += multi_label_loss
        x, prev = _get_rnn_x_prev(labels, x, current, params.mlp_sizes[-1], params)
        keep_labels = _get_updated_labels(labels)
        unique_rating = tf.constant(1, dtype=tf.float32)
        shape_invariants = [unique_rating.get_shape(), tf.TensorShape([None, 1]), \
        tf.TensorShape([None, params.mlp_sizes[-1]]), tf.TensorShape([1, params.mlp_sizes[-1]]), \
        permutation_loss.get_shape()]

        def _cond(unique_rating, keep_labels, \
            x, prev, permutation_loss):
            return tf.greater(tf.reduce_max(keep_labels), mini_label)

        def _gen_loop_body():
            def loop_body(unique_rating, keep_labels, \
                x, prev, permutation_loss):
                tiled_shape = tf.shape(keep_labels)
                prev = tf.ones(tiled_shape) * prev
                # ls, current = gru(x, prev, params)
                if params.rnn == 'C1':
                    ls, current = rnnC1(x, prev, params)
                else:
                    ls, current = rnnC2(x, prev, params)
                total_predicted_scores = ls - tf.reduce_min(ls)
                # total_predicted_scores = ls
                multi_label_loss = _get_multi_label_loss(keep_labels, total_predicted_scores)
                permutation_loss += multi_label_loss
                # update
                x, prev = _get_rnn_x_prev(keep_labels, x, current, params.mlp_sizes[-1], params)
                keep_labels = _get_updated_labels(keep_labels)
                return tf.add(unique_rating, 1.0), keep_labels, \
                x, prev, permutation_loss
            return loop_body

        unique_rating, keep_labels, \
        x, prev, permutation_loss = tf.while_loop(_cond, _gen_loop_body(),
            [unique_rating, keep_labels, \
            x, prev, permutation_loss], shape_invariants=shape_invariants)
        permutation_loss /= unique_rating
    return total_predicted_scores, permutation_loss

# urRank
def rnnC1(x, hprev, params):
    with tf.variable_scope('RNNC1', reuse=tf.AUTO_REUSE):
        # initializer
        xav_init = tf.contrib.layers.xavier_initializer()
        # params
        W = tf.get_variable('W', shape=[params.mlp_sizes[-1], params.mlp_sizes[-1]], \
            initializer=xav_init)
        U = tf.get_variable('U', shape=[params.mlp_sizes[-1], params.mlp_sizes[-1]], \
            initializer=xav_init)
        b = tf.get_variable('b', shape=[params.mlp_sizes[-1]], \
            initializer=tf.constant_initializer(0.))
        # b2 = tf.get_variable('b2', shape=[1], \
        #     initializer=tf.constant_initializer(0.))
        LW = tf.get_variable('LW', shape=[params.mlp_sizes[-1], 1], \
            initializer=xav_init)
        # current hidden state
        h = tf.tanh(tf.matmul(hprev, W) + tf.matmul(x, U) + b)
        # ls = tf.matmul(h, LW) + b2
        ls = tf.matmul(h, LW)
    return ls, h

# standard rnn
def rnnC2(x, hprev, params):
    with tf.variable_scope('RNNC2', reuse=tf.AUTO_REUSE):
        # initializer
        xav_init = tf.contrib.layers.xavier_initializer()
        # params
        W = tf.get_variable('W', shape=[params.mlp_sizes[-1], params.mlp_sizes[-1]], \
            initializer=xav_init)
        U = tf.get_variable('U', shape=[params.mlp_sizes[-1], params.mlp_sizes[-1]], \
            initializer=xav_init)
        b = tf.get_variable('b', shape=[params.mlp_sizes[-1]], \
            initializer=tf.constant_initializer(0.))

        LW = tf.get_variable('LW', shape=[params.mlp_sizes[-1] * 2, 1], \
            initializer=xav_init)
        # current hidden state
        h = tf.tanh(tf.matmul(hprev, W) + tf.matmul(x, U) + b)
        ls = tf.matmul(tf.concat([h, x], 1), LW)
    return ls, h

def _get_rnn_x_prev(labels, x, current, rnn_state_size, params):
    true_label = tf.squeeze(tf.reduce_max(labels))
          
    keep_indices = tf.where(tf.not_equal(tf.squeeze(labels), true_label))
    x = tf.gather_nd(x, keep_indices)
    x = tf.reshape(x, [-1, rnn_state_size])
    # new ht_1 ~~ average of sts and cts at ok_label_indices
    # NEEDS THE squeeze
    ok_label_indices = tf.where(tf.equal(tf.squeeze(labels), true_label))
    # ht
    keep_hts = tf.gather_nd(current, ok_label_indices)
    if params.pooling == 'MP':
        # max
        ht_1 = tf.reduce_max(keep_hts, keepdims=True, axis=0)
    elif params.pooling == 'AP':
        # mean
        ht_1 = tf.reduce_mean(keep_hts, keepdims=True, axis=0)
    else:
        ht_1 = tf.reduce_max(keep_hts, keepdims=True, axis=0)
    ht_1 = tf.reshape(ht_1, [1, rnn_state_size])
    return x, ht_1

def gru(x_t, hprev, params):
    with tf.variable_scope('GRU', reuse=tf.AUTO_REUSE):
        xav_init = tf.contrib.layers.xavier_initializer()
        U = tf.get_variable('U', shape=[4, params.mlp_sizes[-1], \
            params.mlp_sizes[-1]], \
            initializer=xav_init)  
        B = tf.get_variable('B', shape=[2, params.mlp_sizes[-1]], \
            initializer=tf.constant_initializer(0.))
        LW = tf.get_variable('LW', shape=[params.mlp_sizes[-1] * 2, 1], \
            initializer=xav_init)
        # try to output similar items in the adjacent positions
        # gather previous internal state and output state
        #  input gate
        z = tf.sigmoid(tf.matmul(x_t, U[0]) + B[0])
        #  forget gate
        r = tf.sigmoid(tf.matmul(x_t, U[1]) + B[1])     
        h_ = tf.tanh(tf.matmul(x_t, U[2]) + tf.matmul(hprev, U[3]) * r)

        h = tf.multiply((1 - z), h_) + tf.multiply(hprev, z)

        ls = tf.matmul(tf.concat([h, x_t], 1), LW)
    return ls, h

def lstm_score(x_t, prev, params):
    with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
        xav_init = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable('W', shape=[4, params.mlp_sizes[-1], \
            params.mlp_sizes[-1]], \
            initializer=xav_init)
        U = tf.get_variable('U', shape=[4, params.mlp_sizes[-1], \
            params.mlp_sizes[-1]], \
            initializer=xav_init)  
        B = tf.get_variable('B', shape=[4, params.mlp_sizes[-1]], \
            initializer=xav_init) 
        LW = tf.get_variable('LW', shape=[params.mlp_sizes[-1], 1], \
            initializer=xav_init)

        st_1, ct_1 = tf.unstack(prev)      
        i = tf.sigmoid(tf.matmul(x_t, U[0]) + tf.matmul(st_1, W[0]) + B[0])
        #  forget gate
        f = tf.sigmoid(tf.matmul(x_t, U[1]) + tf.matmul(st_1, W[1]) + B[1])
        #  output gate
        o = tf.sigmoid(tf.matmul(x_t, U[2]) + tf.matmul(st_1, W[2]) + B[2])
        #  gate weights
        c_ = tf.tanh(tf.matmul(x_t, U[3]) + tf.matmul(st_1, W[3]) + B[3])
        # new internal cell state or say state or context
        ct = ct_1 * f + c_ * i
        # new output (state) or say ht
        st = tf.tanh(ct) * o
        current = tf.stack([st, ct])
        ls = tf.matmul(st, LW)
    return ls, current

def _get_multi_label_loss(keep_labels, keep_predicted_scores):
    true_label = tf.squeeze(tf.reduce_max(keep_labels))
    ok_label_indices = tf.where(tf.equal(keep_labels, true_label))
    should_increase_predicted_scores = tf.gather_nd(keep_predicted_scores, ok_label_indices)
    exp_should_increase_predicted_scores = tf.exp(should_increase_predicted_scores) 
    exp_keep_predicted_scores = tf.exp(keep_predicted_scores)
    left = exp_should_increase_predicted_scores + tf.reduce_sum(exp_keep_predicted_scores) - \
            tf.reduce_sum(exp_should_increase_predicted_scores)
    # ln 
    multi_label_loss = tf.reduce_sum(should_increase_predicted_scores - tf.log(left))
    multi_label_loss *= -(2**true_label - 1)
    return multi_label_loss    

def _get_ur_logits(features, params):
    # mlp for rnn
    with tf.variable_scope('ur', reuse=tf.AUTO_REUSE):
        # mlp for rnn
        out_l0 = tf.layers.dense(features, params.mlp_sizes[0], name='full_dense_l0', activation=tf.nn.relu)
        out_l1 = tf.layers.dense(out_l0, params.mlp_sizes[-1], name='full_dense_l1', activation=tf.nn.relu)
        # dropout or batch normalization did not help
        # out_l1 = tf.nn.dropout(out_l1, params.dropout_rate)
        x = tf.squeeze(out_l1)
        x = tf.reshape(x, [-1, params.mlp_sizes[-1]])
        h_1 = tf.zeros(tf.shape(x))
        h_1 = tf.cast(h_1, dtype=tf.float32)
        # ls, current = gru(x, h_1, params)
        if params.rnn == 'C2':
            ls, current = rnnC2(x, h_1, params)
        else:
            ls, current = rnnC1(x, h_1, params)
    return ls

def _get_updated_labels(keep_labels):
    keep_indices = tf.where(tf.not_equal(keep_labels, tf.reduce_max(keep_labels)))
    keep_labels = tf.gather_nd(keep_labels, keep_indices)
    # keep/check the shape [-1, 1]
    keep_labels = tf.reshape(keep_labels, [-1, 1])   
    return keep_labels

def _get_updated_predictions_labels(keep_labels, keep_predicted_scores):
    keep_indices = tf.where(tf.not_equal(keep_labels, tf.reduce_max(keep_labels)))
    keep_predicted_scores = tf.gather_nd(keep_predicted_scores, keep_indices)
    keep_labels = tf.gather_nd(keep_labels, keep_indices)
    # keep/check the shape [-1, 1]
    keep_predicted_scores = tf.reshape(keep_predicted_scores, [-1, 1])
    keep_labels = tf.reshape(keep_labels, [-1, 1])   
    return keep_labels, keep_predicted_scores

def _get_total_predictions(logits, ls):
    predicted_scores = tf.reshape(logits, [-1, 1])
    total_predicted_scores = predicted_scores
    if ls is not None:
        total_predicted_scores = predicted_scores + tf.reshape(ls, [-1, 1])
    total_predicted_scores = total_predicted_scores - tf.reduce_min(total_predicted_scores)   
    return total_predicted_scores, predicted_scores 

def _get_updates(keep_labels, keep_predicted_scores, x, st, ct, rnn_state_size):
    keep_predicted_scores = tf.reshape(keep_predicted_scores, [-1, 1])
    keep_labels = tf.reshape(keep_labels, [-1, 1])
    true_label = tf.squeeze(tf.reduce_max(keep_labels))

    keep_indices = tf.where(tf.not_equal(tf.squeeze(keep_labels), true_label))

    keep_predicted_scores = tf.gather_nd(keep_predicted_scores, keep_indices)
    keep_labels = tf.gather_nd(keep_labels, keep_indices)
    # keep/check the shape [-1, 1]
    keep_predicted_scores = tf.reshape(keep_predicted_scores, [-1, 1])
    keep_labels = tf.reshape(keep_labels, [-1, 1])
    # keep_x
    
    x = tf.gather_nd(x, keep_indices)
    x = tf.reshape(x, [-1, rnn_state_size])
    # new st_1 ct_1 ~~ average of sts and cts at ok_label_indices
    true_label = tf.squeeze(tf.reduce_max(keep_labels))
    # NEEDS THE squeeze
    ok_label_indices = tf.where(tf.equal(tf.squeeze(keep_labels), true_label))
    # st
    keep_sts = tf.gather_nd(st, ok_label_indices)
    # st_1 = tf.reduce_max(keep_sts, keepdims=True, axis=0)
    st_1 = tf.reduce_mean(keep_sts, keepdims=True, axis=0)
    st_1 = tf.reshape(st_1, [1, rnn_state_size])
    # same for ct
    keep_cts = tf.gather_nd(ct, ok_label_indices)
    # ct_1 = tf.reduce_max(keep_cts, keepdims=True, axis=0)
    ct_1 = tf.reduce_mean(keep_cts, keepdims=True, axis=0)
    ct_1 = tf.reshape(ct_1, [1, rnn_state_size])
    return keep_labels, keep_predicted_scores, x, st_1, ct_1

def _get_actions_ratings(keep_labels, total_predicted_scores, iteration, is_training):
    # take all indices that give the current max label scores during training
    true_label = tf.squeeze(tf.reduce_max(keep_labels))
    ok_label_indices = tf.where(tf.equal(tf.squeeze(keep_labels), true_label))
    # take the only one index that gives the current prediction score during validation and inference
    prediction_index = tf.argmax(total_predicted_scores, axis=0)    
    if is_training:
        # keep_labels [-1, 1]
        # take all indices
        action_index = ok_label_indices
    else:
        # take the only one index
        action_index = prediction_index
    action_index = tf.cast(action_index, tf.int32)
    action_index = tf.reshape(action_index, [-1, 1])
    if is_training:
        # take all indices
        rating = tf.gather_nd(keep_labels, action_index)
        # action = rating
    else:
        # take the only one index
        # action = tf.cast(iteration, dtype=tf.float32)
        rating = tf.gather(keep_labels, action_index)
    return rating

def _get_rnn_leave_one_predictions_labels(keep_labels, keep_predicted_scores, 
    x, current, n_data, action_index, rnn_state_size):
    action_index = tf.cast(action_index, tf.int32)
    indices = tf.range(n_data, dtype=tf.int32)
    keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_index)))
    keep_predicted_scores = tf.gather_nd(keep_predicted_scores, keep_indices)
    keep_labels = tf.gather_nd(keep_labels, keep_indices)
    # keep/check the shape [-1, 1]
    keep_predicted_scores = tf.reshape(keep_predicted_scores, [-1, 1])
    keep_labels = tf.reshape(keep_labels, [-1, 1])
    # new x ht
    x = tf.gather_nd(x, keep_indices)
    ht_1 = tf.gather(current, tf.reshape(action_index, [-1, 1]))
    ht_1 = tf.reshape(ht_1, [1, rnn_state_size])   
    return keep_labels, keep_predicted_scores, x, ht_1

def _get_leave_one_predictions_labels(keep_labels, keep_predicted_scores, 
    x, current, n_data, action_index, rnn_state_size):
    action_index = tf.cast(action_index, tf.int32)
    indices = tf.range(n_data, dtype=tf.int32)
    keep_indices = tf.where(tf.not_equal(indices, tf.squeeze(action_index)))
    keep_predicted_scores = tf.gather_nd(keep_predicted_scores, keep_indices)
    keep_labels = tf.gather_nd(keep_labels, keep_indices)
    # keep/check the shape [-1, 1]
    keep_predicted_scores = tf.reshape(keep_predicted_scores, [-1, 1])
    keep_labels = tf.reshape(keep_labels, [-1, 1])
    # new x st ct
    st, ct = tf.unstack(current)
    x = tf.gather_nd(x, keep_indices)
    st_1 = tf.gather(st, tf.reshape(action_index, [-1, 1]))
    st_1 = tf.reshape(st_1, [1, rnn_state_size])
    ct_1 = tf.gather(ct, tf.reshape(action_index, [-1, 1]))
    ct_1 = tf.reshape(ct_1, [1, rnn_state_size])
    prev = tf.stack([st_1, ct_1])    
    return keep_labels, keep_predicted_scores, x, prev

def _get_lstm_x_prev(labels, x, current, rnn_state_size):
    true_label = tf.squeeze(tf.reduce_max(labels))
          
    keep_indices = tf.where(tf.not_equal(tf.squeeze(labels), true_label))
    x = tf.gather_nd(x, keep_indices)
    x = tf.reshape(x, [-1, rnn_state_size])
    # new st_1 ct_1 ~~ average of sts and cts at ok_label_indices
    # NEEDS THE squeeze
    ok_label_indices = tf.where(tf.equal(tf.squeeze(labels), true_label))
    # st
    st, ct = tf.unstack(current)
    keep_sts = tf.gather_nd(st, ok_label_indices)

    # st_1 = tf.reduce_max(keep_sts, keepdims=True, axis=0)
    st_1 = tf.reduce_mean(keep_sts, keepdims=True, axis=0)
    st_1 = tf.reshape(st_1, [1, rnn_state_size])
    # same for ct
    keep_cts = tf.gather_nd(ct, ok_label_indices)
    # ct_1 = tf.reduce_max(keep_cts, keepdims=True, axis=0)
    ct_1 = tf.reduce_mean(keep_cts, keepdims=True, axis=0)
    ct_1 = tf.reshape(ct_1, [1, rnn_state_size])
    prev = tf.stack([st_1, ct_1])
    return x, prev

# LSTM based
def build_gl_LSTM_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! when using the build_model mdprank needs a learning_rate around 1e-5 - 1e-7
    """
    permutation_loss = tf.constant(0, dtype=tf.float32)
    features = inputs['features']
    features = tf.reshape(features, [-1, int(params.feature_dim)])
    # logits = _get_mlp_logits(features, params)
    # mlp for lstm
    out_l0 = tf.layers.dense(features, params.mlp_sizes[0], name='full_dense_l0', activation=tf.nn.relu)
    out_1 = tf.layers.dense(out_l0, params.mlp_sizes[-1], name='full_dense_l1', activation=tf.nn.relu)
    out_1 = tf.squeeze(out_1)
    out_1 = tf.reshape(out_1, [-1, params.mlp_sizes[-1]])
    st_1 = tf.zeros(tf.shape(out_1))
    st_1 = tf.cast(st_1, dtype=tf.float32)
    ct_1 = tf.zeros(tf.shape(out_1))
    ct_1 = tf.cast(ct_1, dtype=tf.float32)
    prev = tf.stack([st_1, ct_1])
    ls, current = lstm_score(out_1, prev, params)
    # global minimum label
    predicted_scores = tf.zeros(tf.shape(ls))
    total_predicted_scores = ls - tf.reduce_min(ls)
    if not is_training:
        # since we use NDCG as metric for the best weights
        # we do not have to calculate permutation_loss
        # permutation_loss is 0 in validation and inference
        return total_predicted_scores, permutation_loss 
    labels = inputs['labels']           
    # total_predicted_scores, predicted_scores = _get_total_predictions(logits, ls)
    n_data = inputs['height']
    # n_data = tf.shape(labels)[0]    
    mini_label = tf.reduce_min(labels)
    actual_top = tf.minimum(params.top_k, tf.squeeze(n_data))
    inputs['no_ndcg'] = True    
    x = out_1
    x = tf.reshape(x, [-1, params.mlp_sizes[-1]])
    multi_label_loss = _get_multi_label_loss(labels, total_predicted_scores)
    permutation_loss += multi_label_loss
    x, prev = _get_lstm_x_prev(labels, x, current, params.mlp_sizes[-1])       
    keep_labels, keep_predicted_scores = _get_updated_predictions_labels(labels, predicted_scores)
    iteration = n_data - 1
    shape_invariants = [iteration.get_shape(), tf.TensorShape([None, 1]), \
    tf.TensorShape([None, params.mlp_sizes[-1]]), tf.TensorShape([2, 1, params.mlp_sizes[-1]]), \
    permutation_loss.get_shape()]

    def _cond(iteration, keep_labels, \
        x, prev, permutation_loss):
        return tf.greater(tf.reduce_max(keep_labels), mini_label)

    def _gen_loop_body():
        def loop_body(iteration, keep_labels, \
            x, prev, permutation_loss):
            st_1, ct_1 = tf.unstack(prev)
            tiled_shape = tf.shape(keep_labels)
            st_1 = tf.ones(tiled_shape) * st_1
            ct_1 = tf.ones(tiled_shape) * ct_1
            prev = tf.stack([st_1, ct_1])
            ls, current = lstm_score(x, prev, params)
            total_predicted_scores = ls - tf.reduce_min(ls)
            # total_predicted_scores, keep_predicted_scores = _get_total_predictions(keep_predicted_scores, ls)         
            multi_label_loss = _get_multi_label_loss(keep_labels, total_predicted_scores)
            permutation_loss += multi_label_loss
            x, prev = _get_lstm_x_prev(keep_labels, x, current, params.mlp_sizes[-1])      
            keep_labels = _get_updated_labels(keep_labels)
            return tf.subtract(iteration, 1), \
            keep_labels, x, prev, permutation_loss
        return loop_body

    iteration, keep_labels, \
    x, prev, permutation_loss = tf.while_loop(_cond, _gen_loop_body(),
        [iteration, keep_labels, \
        x, prev, permutation_loss], shape_invariants=shape_invariants)  
    permutation_loss /= tf.cast(tf.squeeze(n_data - iteration), dtype=tf.float32)
    return predicted_scores, permutation_loss

def equal_rating_query(label_shape):
    permutation_loss = tf.constant(0.0, dtype=tf.float32)
    sudo_actions = tf.fill(label_shape, 0.0)
    return sudo_actions, permutation_loss

def build_model(is_training, inputs, params, weak_learner_id):
    """Compute logits of the model
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! when using the build_model mdprank needs a learning_rate around 1e-5 - 1e-7
    """
    if params.use_residual:
        return build_residual_model(is_training, inputs, \
            params, weak_learner_id)
    if params.loss_fn == 'rlrank':
        return build_rl_model(is_training, inputs, params)
    if params.loss_fn == 'urrank':
        if params.rnn == 'LSTM':
            return build_gl_LSTM_model(is_training, inputs, params)
        else: 
            return build_ur_model(is_training, inputs, params)
    if params.loss_fn == 'urank':
        return build_u_model(is_training, inputs, params)
    if params.loss_fn == 'grank':
        return build_g_model(is_training, inputs, params)
    # other loss functions
    permutation_loss = tf.constant(0, dtype=tf.float32)
    features = inputs['features']
    features = tf.reshape(features, [-1, int(params.feature_dim)])
    logits = _get_mlp_logits(features, params)
    # best try
    # mdprank
    if params.loss_fn == 'mdprank':
        if is_training:
            # inputs['labels'], logits = sample.softmax_sample(inputs['labels'], logits)
            if random.random() < params.exploration:
                # random
                pass
            else:
                inputs['labels'], logits = sample.softmax_sample(inputs['labels'], logits)
                # # the max action
                # inputs['labels'], logits = math_fns.get_logit_orders(inputs['labels'], logits)
        else:
            # # the max action
            inputs['labels'], logits = math_fns.get_logit_orders(inputs['labels'], logits)
    return logits, permutation_loss

def model_fn(mode, inputs, params, reuse=False, weak_learner_id=0):
    """Model function defining the graph operations.
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights
    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    is_test = (mode == 'test')
    weak_learner_id = int(weak_learner_id)
    # test will calculate NDCG and ERR directly
    # !!! (for real application please add constraints)
    labels = inputs['labels']
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        predictions, permutation_loss = build_model(is_training, inputs, params, \
                weak_learner_id=weak_learner_id)
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        if is_training:
            loss = get_loss(predictions, labels, params, permutation_loss)
            if params.use_regularization:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                loss += tf.reduce_sum(reg_losses)
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, params.gradient_clip_value)
            train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        ndcg_top_ks =  params.top_ks
        use_err_metric = False
        use_loss_metrics = False
        # loss = tf.Print(loss, [loss], message="The loss is : \n", summarize=200)    
        if is_training:
            use_loss_metrics = True
        if is_test:
            # do not report err in training or validation
            # for fast training
            # report err in test
            use_err_metric = True
        if 'no_ndcg' in inputs:
            use_ndcg_metrics = False
            metrics = search_metrics.get_search_metric_fn(labels, predictions, \
                ndcg_top_ks=ndcg_top_ks, \
                use_binary_metrics=False, use_err_metric=use_err_metric, \
                use_ndcg_metrics=use_ndcg_metrics)
        else:
            metrics = search_metrics.get_search_metric_fn(labels, predictions, \
                ndcg_top_ks=ndcg_top_ks, use_binary_metrics=False, use_err_metric=use_err_metric)        
        if use_loss_metrics == True:
            # Summaries for training
            tf.summary.scalar('loss', loss)            
            # metrics.update(loss_metric)
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec["predictions"] = predictions
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    if is_training:
        model_spec['train_op'] = train_op
        model_spec['loss'] = loss
    return model_spec

def get_loss(predicted_scores, labels,
             params, permutation_loss=None):
    """ 
    Return loss based on loss_function_str
    Note: this is for models that have real loss functions
    """

    def _ranknet():
        pairwise_predicted_scores = scores.get_pairwise_scores(predicted_scores)
        pairwise_label_scores = scores.get_pairwise_label_scores(labels)

        loss = loss_fns.get_pair_loss(pairwise_label_scores, pairwise_predicted_scores,
          params)
        return loss

    def _softmax_ranknet():
        pairwise_predicted_scores = scores.get_softmax_pairwise_scores(predicted_scores)
        pairwise_label_scores = scores.get_pairwise_label_scores(labels)

        loss = loss_fns.get_pair_loss(pairwise_label_scores, pairwise_predicted_scores,
          params)
        return loss

    def _listnet():
        return loss_fns.get_listnet_loss(labels, predicted_scores)

    def _attrank():
        return loss_fns.get_attrank_loss(labels, predicted_scores, weights)

    def _listmle():
        return loss_fns.get_listmle_loss(labels, predicted_scores)

    def _pointwise():
        return loss_fns.get_pointwise_loss(labels, predicted_scores)

    def _ranksvm():
        pairwise_predicted_scores = scores.get_pairwise_scores(predicted_scores)
        pairwise_label_scores = scores.get_pairwise_label_scores(labels)        
        return loss_fns.get_hinge_loss(pairwise_label_scores, pairwise_predicted_scores,
          params)

    def _lambdarank():
        pairwise_predicted_scores = scores.get_pairwise_scores(predicted_scores)
        pairwise_label_scores = scores.get_pairwise_label_scores(labels)
        n_data = tf.shape(labels)[0]
        swapped_ndcg = math_fns.cal_swapped_ndcg(labels,
          predicted_scores, top_k_int=n_data)
        loss = loss_fns.get_lambda_pair_loss(pairwise_label_scores, pairwise_predicted_scores,
          params, swapped_ndcg)
        return loss

    def _urank():
        return permutation_loss

    def _grank():
        return permutation_loss

    def _urrank():
        return permutation_loss

    def _urrank():
        return permutation_loss

    def _residual():
        return permutation_loss

    def _rlrank_pair():
        pairwise_predicted_scores = scores.get_pairwise_scores(predicted_scores)
        pairwise_label_scores = scores.get_rl_pairwise_label_scores(labels)        
        loss = loss_fns.get_rlrank_pair_loss(pairwise_label_scores, pairwise_predicted_scores, params)
        return loss

    def _mdprank():
        return loss_fns.get_mdprank_loss(labels, predicted_scores)

    options = {'ranknet': _ranknet,
            'softmax_ranknet': _softmax_ranknet,
            'listnet': _listnet,
            'attrank': _attrank,
            'listmle': _listmle,
            'pointwise': _pointwise,
            'lambdarank': _lambdarank,
            'pointwise_baseline': _pointwise,
            'ranksvm': _ranksvm,
            'mdprank': _mdprank,
            'urank': _urank,
            'grank': _grank,            
            'urrank': _urrank,
            'residual': _residual,
    }
    loss_function_str = params.loss_fn

    return options[loss_function_str]()