from tensorflow.python.framework import ops
import tensorflow as tf
import utils
import numpy as np

class DeepWordMatchModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def _set_params(self, params):
        self.params = params
        self.n = int(params.max_query_len*(params.max_ad_len/2))

    def _set_mode(self, mode):
        self.mode = mode
        self.is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

    def _temp_conv(self, inputs, conv_word_size, num_filters, filter_size=3, non_linear=None, scope=None):
        with tf.variable_scope(scope or "temp_conv"):
            # [batch_size, max_seq_len, embedding_size]
            kernel1 = tf.get_variable("conv_w", [filter_size, 
                                                 conv_word_size,
                                                 num_filters])

            #[batch_size, max_seq_len, num_filters]
            conv_outs = tf.nn.conv1d(inputs, 
                                     kernel1, 
                                     stride=1, 
                                     padding='SAME')

            #[batch_sixe, max_seq_len, num_filters]
            if non_linear:
                conv_outs = non_linear(conv_outs)
            return conv_outs
         
    def _conv_block(self, inputs, conv_word_size, num_filters, filter_size=3, scope=None):
        
        with tf.variable_scope(scope or "conv_block"):
            #conv 
            kernel1 = tf.get_variable("kernel_1", [filter_size,
                                                  conv_word_size,
                                                  num_filters])
            #[batch_size, inputs.shape[1],  num_filters] 
            conv_outputs = tf.nn.conv1d(inputs, 
                                        kernel1, 
                                        stride=1, 
                                        padding='SAME')
            # batch normalization
            conv_outputs = tf.layers.batch_normalization(conv_outputs, self.is_training)
            #print("c:",conv_outputs)
            # relu
            conv_outputs = tf.nn.relu(conv_outputs)

            kernel2 = tf.get_variable("kernel_2", [filter_size,
                                                   num_filters,
                                                   num_filters])
            conv_outputs = tf.nn.conv1d(conv_outputs, 
                                        kernel2, 
                                        stride=1,  
                                        padding='SAME')
            #batch normalization
            conv_outputs = tf.layers.batch_normalization(conv_outputs, self.is_training)

            # relu
            conv_outputs = tf.nn.relu(conv_outputs)

            return conv_outputs
    
    def _max_pool(self, inputs, k, scope=None):
        with tf.variable_scope(scope or "maxPool") as scope:
            inputs = tf.expand_dims(inputs, axis=-1)

            #[batch_size, max_seq_len/k, inputs[2]]
            max_pool_outputs = tf.nn.max_pool(inputs,
                                              ksize=[1,k,1,1],
                                              strides=[1,k,1,1],
                                              padding='SAME')
            return tf.squeeze(max_pool_outputs, axis=-1)
                           
                                
    def _build_graph(self):
        # self.query_ids 
                                    
        #embedding
	#one_hot vector: [batch_size, max_sentence_len, one_hot_depth]
        with tf.variable_scope("embedding") as scope:
            embedding = tf.get_variable('embedding_matrix',
                                        initializer=tf.random_uniform([self.vocab_size,
                                                                      self.params.embedding_size]),
                                        dtype=tf.float32)

            embedded_query_ids = tf.nn.embedding_lookup(embedding, self.query_ids)

            embedded_ad_ids = tf.nn.embedding_lookup(embedding, self.ad_ids)

        with tf.variable_scope("tempConvQuery") as scope:
            query_conv_outputs = self._temp_conv(embedded_query_ids, 
                                                 self.params.embedding_size,
                                                 self.params.conv_word_size, 
                                                 scope=scope)
            print(query_conv_outputs)
            query_conv_outputs = self._conv_block(query_conv_outputs, 
                                                  self.params.conv_word_size,
                                                  self.params.conv_word_size,
                                                  scope=scope)
            print(query_conv_outputs)

        with tf.variable_scope("tempConvAd") as scope:
            ad_conv_outputs = self._temp_conv(embedded_ad_ids,
                                             self.params.embedding_size,
                                             self.params.conv_word_size,
                                             scope=scope)
            print("ad:", ad_conv_outputs)
            ad_conv_outputs = self._conv_block(ad_conv_outputs, 
                                               self.params.conv_word_size,
                                               self.params.conv_word_size, 
                                               scope=scope)
            ad_conv_outputs = self._max_pool(ad_conv_outputs, 2, scope=scope)
            print("ad:", ad_conv_outputs)

        with tf.variable_scope("crossConvOp") as scope:
            #conv_outputs = tf.py_func(self._cross_conv_op,
            #                          [ad_conv_outputs, query_conv_outputs],
            #                          [tf.float32])
            #with ops.name_scope("crossConvolutionOp",  values=[query_conv_outputs, ad_conv_outputs]) as name:
            #    conv_outputs = utils.py_func(self._cross_conv_op, 
            #                                 [query_conv_outputs, ad_conv_outputs],
            #                                 [tf.float32],
            #                                 name=name,
            #                                 grad=self._cross_conv_op_grad)
            conv_outputs = self._cross_conv_op(query_conv_outputs, 
                                               ad_conv_outputs)

            #conv_outputs = tf.convert_to_tensor(conv_outputs)
            #conv_outputs = tf.reshape(conv_outputs, [self.batch_size,
            #                                         self.n,
            #                                         self.params.conv_word_size*2])
            print(conv_outputs)
            conv_outputs = self._temp_conv(conv_outputs, 
                                           self.params.conv_word_size*2, 
                                           self.params.conv_word_size, 
                                           non_linear=tf.nn.relu,
                                           scope=scope)
            print(conv_outputs)
            conv_outputs = self._max_pool(conv_outputs, 4, scope=scope)
            print(conv_outputs)

        with tf.variable_scope("finalBloc1") as scope:
            conv_outputs = self._conv_block(conv_outputs, 
                                            self.params.conv_word_size,
                                            self.params.conv_word_size*2,
                                            scope=scope)
            conv_outputs = self._max_pool(conv_outputs, 4, scope=scope)

        with tf.variable_scope("finalBlock2") as scope:
            conv_outputs = self._conv_block(conv_outputs,
                                            self.params.conv_word_size*2,
                                            self.params.conv_word_size*2,
                                            scope=scope)
            conv_outputs = self._max_pool(conv_outputs, 4, scope=scope)
            print(conv_outputs)

        with tf.variable_scope("finalBlock3") as scope:
            flat_dim = [self.batch_size,
                        self.n*2]
            flat_outputs = tf.reshape(conv_outputs, flat_dim)
            print(flat_outputs)
            fc_outputs = tf.contrib.layers.fully_connected(flat_outputs, 
                                                          self.params.fc_output_size)
            fc_outputs = tf.contrib.layers.fully_connected(fc_outputs,
                                                           self.params.fc_output_size)
            self.logits = tf.contrib.layers.fully_connected(fc_outputs,
                                                            1,
                                                            activation_fn=None)
            print(self.logits)
            self.preds = tf.nn.sigmoid(self.logits)
            
    def _add_train_layer(self):
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            with tf.variable_scope("trainLayer") as scope:
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                            logits=self.logits))
                #print("losses:", self.losses)
                #self.loss = tf.reduce_mean(self.losses)
                
                if self.mode == tf.estimator.ModeKeys.TRAIN:
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.params.learning_rate)
                    self.train_op = optimizer.minimize(
                        self.loss, 
                        global_step=tf.contrib.framework.get_global_step())

    def _cross_conv_op(self, query_conv_outputs, ad_conv_outputs):
        query_size = int(query_conv_outputs.get_shape()[1])
        ad_size = int(ad_conv_outputs.get_shape()[1])
        extended_query_conv_outputs = tf.concat([query_conv_outputs]*ad_size, axis=1)
        extended_ad_conv_outputs = tf.concat([ad_conv_outputs]*query_size, axis=1)
        cross_conv_outputs = tf.concat([extended_query_conv_outputs,
                                        extended_ad_conv_outputs], axis=-1)
        return cross_conv_outputs
        
    def _get_predictions(self):
        predictions = {"clicked": self.preds }
        return predictions

    def _set_placeholders(self, features, labels):
        self.batch_size = tf.shape(features["query"])[0]

        self.query_ids = features["query"]
        self.query_ids = tf.reshape(self.query_ids, [self.batch_size,
                                                     self.params.max_query_len])
        self.ad_ids = features["ad"]
        self.ad_ids = tf.reshape(self.ad_ids, [self.batch_size,
                                               self.params.max_ad_len])

        self.labels = tf.to_float(labels)

    def create_model_fn(self):
        def model_fn(features, labels, params, mode):
            self._set_params(params)
            self._set_mode(mode)
            self._set_placeholders(features, labels)
            self._build_graph()
            self._add_train_layer()
            
            predictions = self._get_predictions()
            
            #print(self.loss)
            eval_metric_ops = {
                "auc": tf.metrics.auc(
                    labels=self.labels, predictions=predictions["clicked"])}
                
            return tf.estimator.EstimatorSpec(mode, 
                                              predictions=predictions, 
                                              loss=self.loss, 
                                              train_op=self.train_op, 
                                              eval_metric_ops=eval_metric_ops)
        return model_fn
