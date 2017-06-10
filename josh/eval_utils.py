from __future__ import absolute_import
from __future__ import division

import os
import json
import random

import tensorflow as tf

from resnet import placesModel
from os.path import join as pjoin
import numpy as np
from scipy import misc
import logging
import matplotlib.pyplot as plt
# from IPython import embed

logging.basicConfig(level=logging.INFO)


def get_orig_val_image(category, first_at=11):
    f = open("data/places2/places365_val.txt")
    cats= {}
    seen = 0
    for line in f:
        tokens = line.split()
        if int(tokens[1]) == category:
            seen +=1 
        if seen == first_at:
            fname = tokens[0]
            f = ("data/places2/val_256/"+fname)
            return f
    return None



def generate_label_map():
    f = open("data/places2/categories_places365.txt")
    cats= {}
    for line in f:
        tokens = line.split()
        cats[int(tokens[1])] = tokens[0]
    return cats

def evaluate(model,sess,examples, correct_limit=5, incorrect_limit=20, correct_filter = [298, 190, 106, 170,2,229], incorrect_filter = [363, 330, 21, 37, 7, 225, 247]):
    #model is an instance of placesModel
    #sess is the tensorflow session
    #examples is a length 2 list of (example_images, example_labels)
    #saliency=tf.gradients(ys=model.label_placeholder,xs=model.input_placeholder)
    #saliency=sess.run(saliency,{model.label_placeholder:examples[1],model.input_placeholder:examples[0],model.is_train_placeholder:False})[0]
    input_feed = {}
    input_feed[model.input_placeholder] = examples[0]
    input_feed[model.label_placeholder] = examples[1]
    input_feed[model.is_train_placeholder]=True
    correct_scores = tf.gather_nd(model.label_predictions,tf.stack((tf.range(examples[0].shape[0]), model.label_placeholder), axis=1))
    saliency=tf.gradients(ys=correct_scores,xs=model.input_placeholder)
    scores = model.answer(sess,examples)
    output_feed = [model.label_predictions,saliency]
    outputs = sess.run(output_feed, input_feed)
    #outputs[0] is a numpy array of shape (num_examples,num_classes)
    #outputs[1][0] is the saliency map numpy array (num_examples, img_height,img_width,num_channels)
    outputs[1][0] = np.max(np.abs(outputs[1][0]), axis=-1)
    print (outputs[0].shape)
    print (outputs[1][0].shape)

    # Uncomment these to get random incorrect/correct images
    #correct_filter = None
    #incorrect_filter = None

    if incorrect_filter != None:
        incorrect_limit = len(incorrect_filter)

    if correct_filter != None:
        correct_limit = len(correct_filter)

    
    
    print(">>><<<<")
    show_orig = True # don't set to false - can't seem to render properly using matplotlib
    cats = generate_label_map()
    correct_count = 0
    incorrect_count = 0
    correct_idxs = []
    incorrect_idxs = []
    for i, l in enumerate(examples[1]):
        correct_label = cats[int(l)]
        predicted_label = cats[int(scores[i])]
        if correct_label != predicted_label:
            print i, correct_label, predicted_label
            incorrect_idxs.append((i,l, correct_label, predicted_label))
        else:
            correct_idxs.append((i,l, correct_label, predicted_label))
    if incorrect_filter == None:
        incorrect_idxs = random.sample(incorrect_idxs, incorrect_limit)
    if correct_filter == None:
        correct_idxs = random.sample(correct_idxs, correct_limit)
    # incorrect pass
    plt.figure(figsize=(3*incorrect_limit,5))
    plt.title("Incorrectly classified" )
    for (i, l, correct_label, predicted_label) in incorrect_idxs:
        if(incorrect_count < incorrect_limit and (incorrect_filter == None or i in incorrect_filter)):
            incorrect_count +=1 
            plt.subplot(2, incorrect_limit, incorrect_count)
            saliency = outputs[1][0][i]
            if show_orig:
                image = misc.imread(get_orig_val_image(int(l)))
                plt.imshow(image)
            else:
                plt.imshow(examples[0][i])
            plt.axis('off')
            correct_label = "/".join(correct_label.split('/')[2:])
            predicted_label = "/".join(predicted_label.split('/')[2:])
            plt.title("true:%s \npred:%s" %( correct_label, predicted_label))
            plt.subplot(2, incorrect_limit, incorrect_limit+incorrect_count)
            plt.imshow(deprocess_image(saliency, rescale=True), cmap=plt.cm.hot)
            plt.axis('off')
        if incorrect_count >= incorrect_limit:
            break

    plt.show()
    # Correct pass
    plt.figure(figsize=(3*correct_limit,5))
    plt.title("Correctly classified" )
    for (i, l, correct_label, predicted_label) in correct_idxs:
        if(correct_count < correct_limit and  (correct_filter == None or i in correct_filter)):
            correct_count +=1 
            plt.subplot(2, correct_limit, correct_count)
            print i, correct_label, predicted_label
            saliency = outputs[1][0][i]
            if show_orig:
                image = misc.imread(get_orig_val_image(int(l)))
                plt.imshow(image)
            else:
                plt.imshow(examples[0][i])
            plt.axis('off')
            correct_label = "/".join(correct_label.split('/')[2:])
            predicted_label = "/".join(predicted_label.split('/')[2:])
            plt.title("%s" %( correct_label))
            plt.subplot(2, correct_limit, correct_limit+correct_count)
            plt.imshow(deprocess_image(saliency, rescale=True), cmap=plt.cm.hot)
            plt.axis('off')
        if correct_count >= correct_limit:
            break

    plt.show()

    print(">>><<<<")

def preprocess_image(img):
    """Preprocess an image for squeezenet.
    
    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


if __name__ == "__main__":
    print "Don't run directly"


