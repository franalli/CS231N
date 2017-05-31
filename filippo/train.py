from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import os
import json

import tensorflow as tf

from resnet import placesModel
from os.path import join as pjoin
import numpy as np
from scipy import misc
import logging

from IPython import embed

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.7, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("input_width", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("input_height", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 1000, "Size of each model hidden layer.")
tf.app.flags.DEFINE_integer("output_size", 365, "The output size of your model.")
tf.app.flags.DEFINE_string("data_dir", "data/places2", "Places directory")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("num_per_class", 0, "Path to store log and flag files ")
tf.app.flags.DEFINE_integer("grad_clip", 1, "whether to clip gradients or not")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("debug",0,"Whether or not to use debug dataset of 10 images per class from val")
tf.app.flags.DEFINE_string("run_name", "32-resnet", "Name to save the .ckpt file")
# tf.app.flags.DEFINE_string("res_stride", 2, "How many conv layers to take before adding in res and resaving")
tf.app.flags.DEFINE_string("l2_reg", 1e2, "L2 regularization strength")


#Should be 18 layer ResNet

layer0=[("batchnorm",1,None,None,True), ("conv",1,(7,7),2,64,  True,False), ("maxpool",1,(3,3), 2,None,None,True,True)]
layer1=[["conv",1,(3,3),1,64,True,False],["conv",1,(3,3),1,64,True,True]]*2
layer2=[["conv",1,(3,3),2,128,True,False],["conv",1,(3,3),1,128,True,True],["conv",1,(3,3),1,128,True,False],["conv",1,(3,3),1,128,True,True]]
layer3=[["conv",1,(3,3),2,256,True,False],["conv",1,(3,3),1,256,True,True],["conv",1,(3,3),1,256,True,False],["conv",1,(3,3),1,256,True,True]]
layer4=[["conv",1,(3,3),2,512,True,False],["conv",1,(3,3),1,512,True,True],["conv",1,(3,3),1,512,True,False],["conv",1,(3,3),1,512,True,True]]
layer5=[("fc",  1,1000,  None, None,True,False),("fc",  1,1000,  None, None,True,False),("fc",  1,365, None, None,False,False)]


#Should be the 34 layer....
# layer0=[("batchnorm",1,None,None,True), ("conv",1,(7,7),(1,2,2,1),64,  True,False), ("maxpool",1,(3,3), 2,None,None,True,True)]
# layer1=[["conv",1,(3,3),(1,1,1,1),64,True,False],["conv",1,(3,3),(1,1,1,1),64,True,True]]*3
# layer2=[["conv",1,(3,3),(1,2,2,1),128,True,False],["conv",1,(3,3),(1,1,1,1),128,True,True]]
# layer2.extend([["conv",1,(3,3),(1,1,1,1),128,True,False],["conv",1,(3,3),(1,1,1,1),128,True,True]]*3)
# layer3=[["conv",1,(3,3),(1,2,2,1),256,True,False],["conv",1,(3,3),(1,1,1,1),256,True,True]]
# layer3.extend([["conv",1,(3,3),(1,1,1,1),256,True,False],["conv",1,(3,3),(1,1,1,1),256,True,True]]*5)
# layer4=[["conv",1,(3,3),(1,2,2,1),512,True,False],["conv",1,(3,3),(1,1,1,1),512,True,True]]
# layer4.extend([["conv",1,(3,3),(1,1,1,1),512,True,False],["conv",1,(3,3),(1,1,1,1),512,True,True]]*2)
# layer5=[("fc",  1,1000,  None,     None,None,False),("fc",  1,365,  None,     None,None,False)]


layer_params=[]
layer_params.extend(layer0)
layer_params.extend(layer1)
layer_params.extend(layer2)
layer_params.extend(layer3)
layer_params.extend(layer4)
layer_params.extend(layer5)
tf.app.flags.DEFINE_integer("layer_params",layer_params,"list of tuples of (type, number,shape,stride,depth,use_batch_norm,add/set residual)")

FLAGS = tf.app.flags.FLAGS

def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs231n-places2-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


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


def initialize_data(file_name,num_per_class=5000):
    if num_per_class == 0:
        num_per_class = 10000000
    print ("LOADING", file_name, "data")
    f=open(FLAGS.data_dir+"/places365_"+file_name+".txt")
    X=[]
    y=[]
    counts={}
    for line in f:
        img_name,img_class=line.strip().split(" ")
        if img_class not in counts:
            counts[img_class]=0
        if counts[img_class]>num_per_class:
            continue

        counts[img_class]+=1
        if file_name=='train':
            img_name = img_name[1:]
        if FLAGS.input_height == 256 and FLAGS.input_width == 256:
            img=misc.imread(FLAGS.data_dir+"/"+file_name+"_256/"+img_name,mode='RGB')
        else:    
            img=misc.imresize(misc.imread(FLAGS.data_dir+"/"+file_name+"_256/"+img_name,mode='RGB'),(FLAGS.input_height,FLAGS.input_width))
        X.append(img)
        y.append(int(img_class))
    return np.array(X),np.array(y)

def preprocess_data(X_train,X_val):
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image.astype(X_val[0][0][0][0])
    return X_train,X_val


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    if FLAGS.debug:
        print ("Doing debug")
        num_in_debug=50
        try:
            arrs=np.load(FLAGS.data_dir+"/debug_"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width)+"_"+str(num_in_debug)+".npz")
            X_train,y_train,X_val,y_val=arrs['X_train'],arrs['y_train'],arrs['X_val'],arrs['y_val']
            print ("Loaded from .npz file")
        except:
            print ("Creating .npz file")
            X,y=initialize_data("val",num_in_debug+1)
            num_classes=np.max(y)+1
            X_train=[]
            X_train=np.zeros((num_classes*num_in_debug,FLAGS.input_height,FLAGS.input_width,3))
            y_train=np.zeros((num_classes*num_in_debug))
            X_val=np.zeros((num_classes,FLAGS.input_height,FLAGS.input_width,3))
            y_val=np.zeros((num_classes))
            for i in range(num_classes):
                cur_X=X[y==i,:,:,:]
                X_train[i*num_in_debug:(i+1)*num_in_debug,:,:,:]=cur_X[:num_in_debug,:,:,:]
                y_train[i*num_in_debug:(i+1)*num_in_debug]=np.zeros((num_in_debug))+i
                X_val[i,:,:,:]=cur_X[num_in_debug,:,:,:]
                y_val[i]=i
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            X_val=np.array(X_val)
            y_val=np.array(y_val)
            np.savez(FLAGS.data_dir+"/debug_"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width)+"_"+str(num_in_debug),X_train=X_train,y_train=y_train,X_val=X_val,y_val=y_val)
    else:
        try:
            arrs=np.load(FLAGS.data_dir+"/full"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width)+"_"+str(FLAGS.num_per_class)+".npz")
            X_train,y_train,X_val,y_val=arrs['X_train'],arrs['y_train'],arrs['X_val'],arrs['y_val']
            print ("Loaded from .npz file")
        except:
            print ("Creating .npz file")
            X_train,y_train=initialize_data("train",FLAGS.num_per_class)
            X_val,y_val=initialize_data("val",FLAGS.num_per_class)
            np.savez(FLAGS.data_dir+"/full"+str(FLAGS.input_height)+"_"+str(FLAGS.input_width)+"_"+str(FLAGS.num_per_class),X_train=X_train,y_train=y_train,X_val=X_val,y_val=y_val)

    print ("X_train",X_train.shape)
    print ("y_train",y_train.shape)
    print ("X_val",X_val.shape)
    print ("y_val",y_val.shape)

    # X_train,X_val = preprocess_data(X_train,X_val)
    train_dataset = [X_train,y_train]
    val_dataset = [X_val,y_val]

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

        model.train(session=sess,
                 train_dataset=train_dataset,
                 val_dataset=val_dataset,
                 train_dir=save_train_dir)

if __name__ == "__main__":
    tf.app.run()

