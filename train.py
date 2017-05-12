from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
import json

import tensorflow as tf

#from places_model import placesModel
from os.path import join as pjoin
import numpy as np
from scipy import misc
import logging

from IPython import embed

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.3, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("input_width", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("input_height", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", nclasses, "The output size of your model.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_integer("output_size", 350, "The output size of your model.")
tf.app.flags.DEFINE_string("data_dir", "data/places2", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("debug",1,"Whether or not to use debug dataset of 10 images per class from val")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_data(file_name):
    print "LOADING", file_name
    f=open(FLAGS.data_dir+"/places365_"+file_name+".txt")
    X=[]
    y=[]
    for line in f:
        img_name,img_class=line.strip().split(" ")
        img=misc.imresize(misc.imread(FLAGS.data_dir+"/"+file_name+"_256/"+img_name,mode="RGB"),(FLAGS.input_height,FLAGS.input_width))
        X.append(img)
        y.append(int(img_class))
    return np.array(X),np.array(y)
def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    train_dataset = initialize_data(train_data_path, keep_as_string=False)
    val_dataset = initialize_data(val_data_path, keep_as_string=False)
    if FLAGS.debug:
        print "Doing debug"
        try:
            arrs=np.load(FLAGS.data_dir+"/debug_"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width)+".npz")
            X_train,y_train,X_val,y_val=arrs['X_train'],arrs['y_train'],arrs['X_val'],arrs['y_val']
            print "Loaded from .npz file"
        except:
            print "Creating .npz file"
            X,y=initialize_data("val")
            num_classes=np.max(y)+1
            X_train=[]
            X_train=np.zeros((num_classes*10,FLAGS.input_height,FLAGS.input_width,3))
            y_train=np.zeros((num_classes*10))
            X_val=np.zeros((num_classes,FLAGS.input_height,FLAGS.input_width,3))
            y_val=np.zeros((num_classes))
            for i in range(num_classes):
                cur_X=X[y==i,:,:,:]
                X_train[i*10:(i+1)*10,:,:,:]=cur_X[:10,:,:,:]
                y_train[i*10:(i+1)*10]=np.zeros((10))+i
                X_val[i,:,:,:]=cur_X[10,:,:,:]
                y_val[i]=i
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            X_val=np.array(X_val)
            y_val=np.array(y_val)
            np.savez(FLAGS.data_dir+"/debug_"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width),X_train=X_train,y_train=y_train,X_val=X_val,y_val=y_val)
    else:
        try:
            arrs=np.load(FLAGS.data_dir+"/full"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width)+".npz")
            X_train,y_train,X_val,y_val=arrs['X_train'],arrs['y_train'],arrs['X_val'],arrs['y_val']
            print "Loaded from .npz file"
        except:
            print "Creating .npz file"
            X_train,y_train=initialize_data("train")
            X_val,y_val=initialize_data("val")
            np.savez(FLAGS.data_dir+"/full"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width),X_train=X_train,y_train=y_train,X_val=X_val,y_val=y_val)

    print "X_train",X_train.shape
    print "y_train",y_train.shape
    print "X_val",X_val.shape
    print "y_val",y_val.shape
    exit()
    data = (X_train,y_train,X_val,y_val)

    model = placesModel(flags=FLAGS)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, model, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        saver = tf.train.Saver()

        qa.train(session=sess,
                 dataset=train_dataset,
                 val_dataset=val_dataset,
                 train_dir=save_train_dir)

if __name__ == "__main__":
    tf.app.run()
