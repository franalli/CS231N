from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from util import Progbar, minibatches

from evaluate import exact_match_score, f1_score

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


        # ==== set up placeholder tokens ========

        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, self.p_size), name='context_placeholder')

        # ==== assemble pieces ====
        with tf.variable_scope("places_model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_system()
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


    def setup_system(self):
        print("model architecture")

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.predictions, self.labels))  


    def optimize(self, session, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
 
        input_feed[self.context_placeholder] = context_batch
        input_feed[self.question_placeholder] = question_batch
        input_feed[self.mask_ctx_placeholder] = mask_ctx_batch
        input_feed[self.mask_q_placeholder] = mask_q_batch
        input_feed[self.dropout_placeholder] = self.flags.dropout
        input_feed[self.answer_span_placeholder] = answer_span_batch

        output_feed = [self.train_op, self.loss]

        _, loss = session.run(output_feed, input_feed)

        return loss

    def decode(self, session, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        input_feed[self.context_placeholder] = context_batch
        input_feed[self.question_placeholder] = question_batch
        input_feed[self.mask_ctx_placeholder] = mask_ctx_batch
        input_feed[self.mask_q_placeholder] = mask_q_batch
        input_feed[self.dropout_placeholder] = self.flags.dropout


        output_feed = [self.start_probs, self.end_probs]

        
        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, data):

        yp_lst = []
        yp2_lst = []
        prog_train = Progbar(target=1 + int(len(data[0]) / self.flags.batch_size))
        for i, batch in enumerate(self.minibatches(data, self.flags.batch_size, shuffle=False)):
            yp, yp2 = self.decode(session, *batch)
            yp_lst.append(yp)
            yp2_lst.append(yp2)
            prog_train.update(i + 1, [("Answering Questions....",0.0)])
        print("")
        yp_all = np.concatenate(yp_lst, axis=0)
        yp2_all = np.concatenate(yp2_lst, axis=0)

        a_s = np.argmax(yp_all, axis=1)
        a_e = np.argmax(yp2_all, axis=1)

        return (a_s, a_e)

    def validate(self, sess, context_batch, question_batch, answer_span_batch, mask_ctx_batch, mask_q_batch):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        input_feed[self.context_placeholder] = context_batch
        input_feed[self.question_placeholder] = question_batch
        input_feed[self.mask_ctx_placeholder] = mask_ctx_batch
        input_feed[self.mask_q_placeholder] = mask_q_batch
        input_feed[self.dropout_placeholder] = self.flags.dropout
        input_feed[self.answer_span_placeholder] = answer_span_batch

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def evaluate_answer(self, session, dataset, context, sample=100, log=False, eval_set='train'):
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
            context=[context[i] for i in inds]
            
        a_s, a_e = self.answer(session, sampled)

        context_ids, question_ids, answer_spans, ctx_mask, q_mask = sampled

        f1 = []
        em = []
        # #embed()
        for i in range(len(sampled[0])):
            pred_words=' '.join(context[i][a_s[i]:a_e[i]+1])
            actual_words=' '.join(context[i][answer_spans[i][0]:answer_spans[i][1]+1])
            f1.append(f1_score(pred_words, actual_words))
            cur_em = exact_match_score(pred_words, actual_words)
            em.append(float(cur_em))

        if log:
            logging.info("{},F1: {}, EM: {}, for {} samples".format(eval_set, np.mean(f1), np.mean(em), sample))

        return np.mean(f1), np.mean(em)

    ### Imported from NERModel
    def run_epoch(self, sess, train_set, val_set, train_context, val_context):
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
                             context=train_context,
                             sample=len(val_set[0]),
                             log=True,
                             eval_set="-Epoch TRAIN-")

        self.evaluate_answer(session=sess,
                             dataset=val_set,
                             context=val_context,
                             sample=None,
                             log=True,
                             eval_set="-Epoch VAL-")

    def train(self, session, dataset, val_dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious approach can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        #self.saver=saver
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        context_ids, question_ids, answer_spans, ctx_mask ,q_mask, train_context = dataset
        train_dataset = [context_ids, question_ids, answer_spans, ctx_mask ,q_mask]
        #train_dataset = [np.array(col) for col in zip(*train_dataset)]

        val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask, val_context = val_dataset
        val_dataset = [val_context_ids, val_question_ids, val_answer_spans, val_ctx_mask, val_q_mask]
        #val_dataset = [np.array(col) for col in zip(*val_dataset)]
        
        num_epochs = self.flags.epochs

        if self.flags.debug:
            train_dataset = [elem[:self.flags.batch_size*1] for elem in train_dataset]
            val_dataset = [elem[:self.flags.batch_size] for elem in val_dataset]
            num_epochs = 100
            #num_epochs = 1

        for epoch in range(num_epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.flags.epochs)
            self.run_epoch(sess=session,
                           train_set=train_dataset, 
                           val_set=val_dataset,
                           train_context=train_context,
                           val_context=val_context)
            logging.info("Saving model in %s", train_dir)
            self.saver.save(session, train_dir+"/"+self.flags.run_name+".ckpt")

    def minibatches(self, data, batch_size, shuffle=True):
        num_data = len(data[0])
        context_ids, question_ids, answer_spans, ctx_mask, q_mask = data
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)
        for minibatch_start in np.arange(0, num_data, batch_size):
            minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
            yield [context_ids[minibatch_indices], question_ids[minibatch_indices], answer_spans[minibatch_indices], ctx_mask[minibatch_indices], q_mask[minibatch_indices]]





