from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import tensorflow.contrib.layers as layers

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
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None,), name='label_placeholder')

        # ==== assemble pieces ====
        with tf.variable_scope("places_model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_graph()
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


    def setup_graph(self):
        with tf.variable_scope("simple_feed_forward",reuse=False):
            W1conv = tf.get_variable('W1conv',shape=[7,7,self.channels,64],initializer=layers.xavier_initializer())
            b1conv = tf.get_variable('b1conv',shape=[64],initializer=tf.zeros_initializer())

            W2conv = tf.get_variable('W2conv',shape=[3,3,64,64],initializer=layers.xavier_initializer())
            b2conv = tf.get_variable('b2conv',shape=[64],initializer=tf.zeros_initializer())
            W3conv = tf.get_variable('W3conv',shape=[3,3,64,64],initializer=layers.xavier_initializer())
            b3conv = tf.get_variable('b3conv',shape=[64],initializer=tf.zeros_initializer())
            W4conv = tf.get_variable('W4conv',shape=[3,3,64,64],initializer=layers.xavier_initializer())
            b4conv = tf.get_variable('b4conv',shape=[64],initializer=tf.zeros_initializer())
            W5conv = tf.get_variable('W5conv',shape=[3,3,64,64],initializer=layers.xavier_initializer())
            b5conv = tf.get_variable('b5conv',shape=[64],initializer=tf.zeros_initializer())

            W6conv = tf.get_variable('W6conv',shape=[3,3,64,128],initializer=layers.xavier_initializer())
            b6conv = tf.get_variable('b6conv',shape=[128],initializer=tf.zeros_initializer())
            W7conv = tf.get_variable('W7conv',shape=[3,3,128,128],initializer=layers.xavier_initializer())
            b7conv = tf.get_variable('b7conv',shape=[128],initializer=tf.zeros_initializer())
            W8conv = tf.get_variable('W8conv',shape=[3,3,128,128],initializer=layers.xavier_initializer())
            b8conv = tf.get_variable('b8conv',shape=[128],initializer=tf.zeros_initializer())
            W9conv = tf.get_variable('W9conv',shape=[3,3,128,128],initializer=layers.xavier_initializer())
            b9conv = tf.get_variable('b9conv',shape=[128],initializer=tf.zeros_initializer())

            W10conv = tf.get_variable('W10conv',shape=[3,3,128,256],initializer=layers.xavier_initializer())
            b10conv = tf.get_variable('b10conv',shape=[256],initializer=tf.zeros_initializer())
            W11conv = tf.get_variable('W11conv',shape=[3,3,256,256],initializer=layers.xavier_initializer())
            b11conv = tf.get_variable('b11conv',shape=[256],initializer=tf.zeros_initializer())
            W12conv = tf.get_variable('W12conv',shape=[3,3,256,256],initializer=layers.xavier_initializer())
            b12conv = tf.get_variable('b12conv',shape=[256],initializer=tf.zeros_initializer())
            W13conv = tf.get_variable('W13conv',shape=[3,3,256,256],initializer=layers.xavier_initializer())
            b13conv = tf.get_variable('b13conv',shape=[256],initializer=tf.zeros_initializer())

            W14conv = tf.get_variable('W14conv',shape=[3,3,256,512],initializer=layers.xavier_initializer())
            b14conv = tf.get_variable('b14conv',shape=[512],initializer=tf.zeros_initializer())
            W15conv = tf.get_variable('W15conv',shape=[3,3,512,512],initializer=layers.xavier_initializer())
            b15conv = tf.get_variable('b15conv',shape=[512],initializer=tf.zeros_initializer())
            W16conv = tf.get_variable('W16conv',shape=[3,3,512,512],initializer=layers.xavier_initializer())
            b16conv = tf.get_variable('b16conv',shape=[512],initializer=tf.zeros_initializer())
            W17conv = tf.get_variable('W17conv',shape=[3,3,512,512],initializer=layers.xavier_initializer())
            b17conv = tf.get_variable('b17conv',shape=[512],initializer=tf.zeros_initializer())

            bn = tf.layers.batch_normalization(self.input_placeholder)
            z1 = tf.nn.conv2d(bn, W1conv,[1,2,2,1],'SAME') + b1conv
            bn1 = tf.layers.batch_normalization(z1)
            h1 = tf.nn.relu(bn1)
            p1 = tf.layers.max_pooling2d(h1,pool_size =(3,3),strides=2)

            z2 = tf.nn.conv2d(p1, W2conv,[1,1,1,1],'SAME') + b2conv
            bn2 = tf.layers.batch_normalization(z2)
            h2 = tf.nn.relu(bn2)

            z3 = tf.nn.conv2d(h2, W3conv,[1,1,1,1],'SAME') + b3conv
            res1 = p1 + z3
            bn3 = tf.layers.batch_normalization(res1)
            h3= tf.nn.relu(bn3)

            z4 = tf.nn.conv2d(h3, W4conv,[1,1,1,1],'SAME') + b4conv
            bn4 = tf.layers.batch_normalization(z4)
            h4= tf.nn.relu(bn4)

            z5 = tf.nn.conv2d(h4, W5conv,[1,1,1,1],'SAME') + b5conv
            res2 = h3 + z5
            bn5 = tf.layers.batch_normalization(res2)
            h5= tf.nn.relu(bn5)

            z6 = tf.nn.conv2d(h5, W6conv,[1,1,1,1],'SAME') + b6conv
            bn6 = tf.layers.batch_normalization(z6)
            h6= tf.nn.relu(bn6)

            z7 = tf.nn.conv2d(h6, W7conv,[1,1,1,1],'SAME') + b7conv
            h5_padded = tf.pad(h5, paddings=([0,0],[0,0], [0, 0],[32,32]), mode='CONSTANT')
            res3 = h5_padded + z7
            bn7 = tf.layers.batch_normalization(res3)
            h7= tf.nn.relu(bn7)

            z8 = tf.nn.conv2d(h7, W8conv,[1,1,1,1],'SAME') + b8conv
            bn8 = tf.layers.batch_normalization(z8)
            h8= tf.nn.relu(bn8)

            z9 = tf.nn.conv2d(h8, W9conv,[1,1,1,1],'SAME') + b9conv
            res4 = h8 + z9
            bn9 = tf.layers.batch_normalization(res4)
            h9= tf.nn.relu(bn9)

            z10 = tf.nn.conv2d(h9, W10conv,[1,1,1,1],'SAME') + b10conv
            bn10 = tf.layers.batch_normalization(z10)
            h10= tf.nn.relu(bn10)

            z11 = tf.nn.conv2d(h10, W11conv,[1,1,1,1],'SAME') + b11conv
            res5 = h10 + z11
            bn11 = tf.layers.batch_normalization(res5)
            h11= tf.nn.relu(bn11)

            z12 = tf.nn.conv2d(h11, W12conv,[1,1,1,1],'SAME') + b12conv
            bn12 = tf.layers.batch_normalization(z12)
            h12= tf.nn.relu(bn12)

            z13 = tf.nn.conv2d(h12, W13conv,[1,1,1,1],'SAME') + b13conv
            res6 = h12 + z13
            bn13 = tf.layers.batch_normalization(res6)
            h13= tf.nn.relu(bn13)

            z14 = tf.nn.conv2d(h13, W14conv,[1,1,1,1],'SAME') + b14conv
            bn14 = tf.layers.batch_normalization(z14)
            h14= tf.nn.relu(bn14)

            z15 = tf.nn.conv2d(h14, W15conv,[1,1,1,1],'SAME') + b15conv
            res7 = h14 + z15
            bn15 = tf.layers.batch_normalization(res7)
            h15= tf.nn.relu(bn15)

            z16 = tf.nn.conv2d(h15, W16conv,[1,1,1,1],'SAME') + b16conv
            bn16 = tf.layers.batch_normalization(z16)
            h16= tf.nn.relu(bn16)

            z17 = tf.nn.conv2d(h16, W17conv,[1,1,1,1],'SAME') + b17conv
            res8 = h16 + z17
            bn17 = tf.layers.batch_normalization(res8)
            h17= tf.nn.relu(bn17)

            p2 = tf.nn.pool(h17,window_shape=(3,3),pooling_type='AVG',padding='VALID')
            flat = layers.flatten(p2)

            W1 = tf.get_variable('W1',shape=[flat.get_shape()[-1],self.h_size],initializer=layers.xavier_initializer())
            b1 = tf.get_variable('b1',shape=[self.h_size],initializer=tf.zeros_initializer())
            W2 = tf.get_variable('W2',shape=[self.h_size,self.flags.output_size],initializer=layers.xavier_initializer())
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
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.label_predictions))


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

        return outputs[0]


    def answer(self, session, data):

        scores = []
        prog_train = Progbar(target=1 + int(len(data[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(data, self.flags.batch_size, shuffle=False)):
            score = self.forward_pass(session, *batch)  
            scores.append(score)
            prog_train.update(i + 1, [("Predicting Images....",0.0)])
        print("")

        predictions = np.argmax(scores, axis=-1)
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

        # print train_dataset[0].shape,train_dataset[1].shape
        # print val_dataset[0].shape,val_dataset[1].shape

        if self.flags.debug:
            train_dataset = [elem[:self.flags.batch_size*1] for elem in train_dataset]
            val_dataset = [elem[:self.flags.batch_size] for elem in val_dataset]
            num_epochs = 100

        # print train_dataset[0].shape,train_dataset[1].shape
        # print val_dataset[0].shape,val_dataset[1].shape
        # assert False

        for epoch in range(num_epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session,
                           train_set=train_dataset, 
                           val_set=val_dataset)
            logging.info("Saving model in %s", train_dir)
            self.saver.save(session, train_dir+"/"+self.flags.run_name+".ckpt")






