from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from util import Progbar, minibatches

# from evaluate import exact_match_score, f1_score

from IPython import embed

from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class placesModel(object):
    def __init__(self, flags):
        """
        Initializes your System

        :param args: pass in more arguments as needed
        """
        self.flags = flags
        self.h_size = self.flags.state_size
        self.dropout = self.flags.dropout
        self.height = self.flags.input_height
        self.width = self.flags.input_width
        self.channels = 3


        # ==== set up placeholder tokens ========

        self.input_placeholder = tf.placeholder(tf.float32, shape=(None,self.height,self.width,self.channels), name='input_placeholder')
        self.label_placeholder = tf.placeholder(tf.int2, shape=(None,), name='label_placeholder')

        # ==== assemble pieces ====
        with tf.variable_scope("places_model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_operations()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = self.flags.learning_rate

        self.learning_rate = self.starter_learning_rate

        # learning rate decay
        # self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
        #                                    1000, 0.96, staircase=True)

        self.optimizer = get_optimizer("adam")
        
        if self.flags.grad_clip:
            # gradient clipping
            self.optimizer = self.optimizer(self.learning_rate)
            grads = self.optimizer.compute_gradients(self.loss)
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, self.flags.max_gradient_norm), var)
            self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)
        else:
            # no gradient clipping
            self.train_op = self.optimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.saver=tf.train.Saver()


    def setup_operations(self):
        with tf.variable_scope("simple_feed_forward",reuse=False):
            W1conv = tf.get_variable('W1conv',shape=[3,3,self.channels,32],initializer=layers.xavier_initializer())
            b1conv = tf.get_variable('b1conv',shape=[32],initializer=tf.zeros_initializer())
            W2conv = tf.get_variable('W2conv',shape=[3,3,32,64],initializer=layers.xavier_initializer())
            b2conv = tf.get_variable('b2conv',shape=[64],initializer=tf.zeros_initializer())
            W3conv = tf.get_variable('W3conv',shape=[3,3,64,64],initializer=layers.xavier_initializer())
            b3conv = tf.get_variable('b3conv',shape=[64],initializer=tf.zeros_initializer())

            z1 = tf.nn.conv2d(self.input_placeholder, W1conv,[1,2,2,1],'SAME') + b1conv
            h1 = tf.nn.relu(z1)
            z2 = tf.nn.conv2d(h1,W2conv,[1,2,2,1],'SAME') + b2conv
            h2 = tf.nn.relu(z2)
            z3 = tf.nn.conv2d(h2,W3conv,[1,1,1,1],'SAME') + b3conv
            h3 = tf.nn.relu(z3)

            flat = layers.flatten(h3)

            W1 = tf.get_variable('W1',shape=[flat.get_shape()[-1],512],initializer=layers.xavier_initializer())
            b1 = tf.get_variable('b1',shape=[512],initializer=tf.zeros_initializer())
            W2 = tf.get_variable('W2',shape=[512,self.flags.output_size],initializer=layers.xavier_initializer())
            b2 = tf.get_variable('b2',shape=[self.flags.output_size],initializer=tf.zeros_initializer())

            z4 = tf.matmul(flat,W1) + b1
            h4 = tf.nn.relu(z4)
            self.label_predictions = tf.matmul(h4,W2) + b2



    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.label_predictions, self.label_placeholder))  


    def optimize(self, session, image_batch, label_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed[self.input_placeholder] = image_batch
        input_feed[self.label_placeholder] = label_batch
        output_feed = [self.train_op, self.loss]
        _, loss = session.run(output_feed, input_feed)

        return loss

    def forward_pass(self, session, image_batch, label_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        input_feed[self.input_placeholder] = image_batch
        input_feed[self.label_placeholder] = label_batch
        output_feed = [self.label_predictions]
        outputs = session.run(output_feed, input_feed)

        return outputs


    def answer(self, session, data):

        scores = []
        prog_train = Progbar(target=1 + int(len(data[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(data, self.flags.batch_size, shuffle=False)):
            score = self.forward_pass(session, *batch)  
            scores.append(label)
            prog_train.update(i + 1, [("Predicting Images....",0.0)])
        print("")
        predictions = np.argmax(scores, axis=1)

        return predictions

    def validate(self, session, image_batch, label_batch):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        input_feed = {}

        input_feed[self.input_placeholder] = image_batch
        input_feed[self.label_placeholder] = label_batch

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def evaluate_answer(self, session, dataset, sample=100, log=False, eval_set='train'):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        if sample is None:
            sampled = dataset
            sample = len(dataset[0])
        else:
            #np.random.seed(0)
            inds = np.random.choice(len(dataset[0]), sample)
            
            sampled = [elem[inds] for elem in dataset]
            
        predictions = self.answer(session, sampled)
        images, labels = sampled
        accuracy = np.mean(predictions == labels)

        if log:
            logging.info("{}, accuracy: {}, for {} samples".format(eval_set, accuracy, sample))

        return accuracy

    def run_epoch(self, sess, train_set, val_set):
        prog_train = Progbar(target=1 + int(len(train_set[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(train_set, self.flags.batch_size)):
            loss = self.optimize(sess, *batch)
            prog_train.update(i + 1, [("train loss", loss)])
        print("")

        #if self.flags.debug == 0:
        prog_val = Progbar(target=1 + int(len(val_set[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(val_set, self.flags.batch_size)):
            val_loss = self.validate(sess, *batch)
            prog_val.update(i + 1, [("val loss", val_loss)])
        print("")

        self.evaluate_answer(session=sess,
                             dataset=train_set,
                             sample=len(val_set[0]),
                             log=True,
                             eval_set="-Epoch TRAIN-")

        self.evaluate_answer(session=sess,
                             dataset=val_set,
                             sample=None,
                             log=True,
                             eval_set="-Epoch VAL-")


    def minibatches(self, data, batch_size, shuffle=True):
        num_data = len(data[0])
        images,labels = data
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, num_data, batch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
            yield [images[minibatch_indices],labels[minibatch_indices]]

    def train(self, session, train_dataset, val_dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious approach can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        #self.saver=saver
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # context_ids, question_ids, answer_spans, ctx_mask ,q_mask, train_context = dataset
        # train_dataset = [context_ids, question_ids, answer_spans, ctx_mask ,q_mask]

        # val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask, val_context = val_dataset
        # val_dataset = [val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask]

        
        num_epochs = self.flags.epochs

        if self.flags.debug:
            train_dataset = [elem[:self.flags.batch_size*1] for elem in train_dataset]
            val_dataset = [elem[:self.flags.batch_size] for elem in val_dataset]
            num_epochs = 10

        for epoch in range(num_epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session,
                           train_set=train_dataset, 
                           val_set=val_dataset)
            logging.info("Saving model in %s", train_dir)
            self.saver.save(session, train_dir+"/"+self.flags.run_name+".ckpt")






