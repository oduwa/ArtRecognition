# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import random, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from skimage import io, transform, util
import skimage
import argparse

# parse cli arguments
ap = argparse.ArgumentParser(description="Tensorflow RNN for text generation")
ap.add_argument('-t', '--train', help = 'Set this flag to train the RNN from scratch', action='store_true', default=False)
ap.add_argument('-t2', '--train2', help = 'Set this flag to train the RNN from the last point', action='store_true', default=False)
ap.add_argument('-c', '--clf', help = 'The file path for the image to be classified', default='')
ap.add_argument('-x', '--test', help = 'Set this flag to apply to test data', action='store_true', default=False)
args = vars(ap.parse_args())
isTrainingPhase = args['train']
isContinuingTraining = args['train2']
isTesting = args['test']
query_file_path = args['clf']

HEIGHT = 96
WIDTH = 96
CHANNELS = 3
IMAGE_DIMS = (HEIGHT, WIDTH, CHANNELS)
N_CLASSES = 4
TARGETS = ['cappella_s_sebastiano','papa_leone', 'pieta', 'porta_santa']

EPOCHS = 301
LEARNING_RATE = 1e-3
BATCH_SIZE = 24

class ExhibitDataset(object):
    def __init__(self, path='dataset'):
        # initialize the data and labels
        self.images = []
        self.labels = []

        # grab the image paths and randomly shuffle them
        image_paths = []
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg")\
                or filename.endswith(".PNG") or filename.endswith(".JPG"):
                    image_paths.append(os.path.join(subdir, filename))

        # Seeding with the same value will give you the same random permutation
        # everytime. Take this out for real shuffle
        random.seed(42)
        random.shuffle(image_paths)
        print('[LOADED {} IMAGES]'.format(len(image_paths)))

        # loop over the input images to properly vectorize them and store labels
        for path in image_paths:
            #print path
            img = io.imread(path)
            img_resized = transform.resize(img, IMAGE_DIMS)
            self.images.append(img_resized)

            label = path.split(os.path.sep)[-2]
            label_vec = np.zeros(N_CLASSES)
            label_vec[TARGETS.index(label)] = 1
            self.labels.append(label_vec)

        # scale the raw pixel intensities to the range [0, 1]
        self.images = np.array(self.images, dtype="float") / 255.0
        self.labels = np.array(self.labels)

        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.images,
        	self.labels, test_size=0.1, random_state=45)

        print("\nTRAIN DATA CLASS DISTRIBUTION:")
        print("cappella_s_sebastiano: {}".format(len([v for v in self.trainY if 'cappella_s_sebastiano' in TARGETS[np.argmax(v)]])))
        print("papa_leone: {}".format(len([v for v in self.trainY if 'papa_leone' in TARGETS[np.argmax(v)]])))
        print("pieta: {}".format(len([v for v in self.trainY if 'pieta' in TARGETS[np.argmax(v)]])))
        print("porta_santa: {}".format(len([v for v in self.trainY if 'porta_santa' in TARGETS[np.argmax(v)]])))

        print("\nTEST DATA CLASS DISTRIBUTION:")
        print("cappella_s_sebastiano: {}".format(len([v for v in self.testY if 'cappella_s_sebastiano' in TARGETS[np.argmax(v)]])))
        print("papa_leone: {}".format(len([v for v in self.testY if 'papa_leone' in TARGETS[np.argmax(v)]])))
        print("pieta: {}".format(len([v for v in self.testY if 'pieta' in TARGETS[np.argmax(v)]])))
        print("porta_santa: {}".format(len([v for v in self.testY if 'porta_santa' in TARGETS[np.argmax(v)]])))

        self.train_data_index = 0
        self.test_data_index = 0

    def next_batch_train(self, batch_size):
        batch_x = np.zeros((batch_size, HEIGHT, WIDTH, CHANNELS),dtype=np.float32)
        batch_y = np.zeros((batch_size, N_CLASSES),dtype=np.float32)

        if self.train_data_index + batch_size >= len(self.trainX):
            self.train_data_index = 0

        #print("FETCHING BATCH FOR IMAGES {} - {}..".format(self.data_index+1, self.data_index + batch_size))

        vec_idx = 0
        for i in range(self.train_data_index,self.train_data_index+batch_size):
            batch_x[vec_idx] = self.trainX[i]
            batch_y[vec_idx] = self.trainY[i]
            vec_idx += 1

        self.train_data_index = (self.train_data_index + batch_size) % len(self.trainX)
        return batch_x,batch_y

    def next_batch_test(self, batch_size):
        batch_x = np.zeros((batch_size, HEIGHT, WIDTH, CHANNELS),dtype=np.float32)
        batch_y = np.zeros((batch_size, N_CLASSES),dtype=np.float32)

        if self.test_data_index + batch_size >= len(self.testX):
            self.test_data_index = 0

        #print("FETCHING BATCH FOR IMAGES {} - {}..".format(self.data_index+1, self.data_index + batch_size))

        vec_idx = 0
        for i in range(self.test_data_index,self.test_data_index+batch_size):
            batch_x[vec_idx] = self.testX[i]
            batch_y[vec_idx] = self.testY[i]
            vec_idx += 1

        self.test_data_index = (self.test_data_index + batch_size) % len(self.testX)

        return batch_x,batch_y

    def train_size(self):
        return len(self.trainX)

    def test_size(self):
        return len(self.testX)

    def get_test_data(self):
        return (self.testX, self.testY)

    def shuffle_train_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.trainX)
        np.random.set_state(rng_state)
        np.random.shuffle(self.trainY)

def prettify_confusion_matrix(cm, ind=['True cappella', 'papa_leone', 'pieta', 'porta_santa'],
 cols=['Pred cappella', 'papa_leone', 'pieta', 'porta_santa']):
    return pd.DataFrame(cm, index=ind, columns=cols)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# tf Graph input. None means that the first dimension can be of any size so it represents the batch size
x = tf.placeholder("float", [None, HEIGHT, WIDTH, CHANNELS])
y_ = tf.placeholder("float", [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32)
is_train = tf.placeholder(tf.bool, name="is_train")

# The first weight tensor has a shape of [5, 5, 1, 32] because our filter is of size
# 5 x 5 (pretty standard). We only have one channel for grayscale and it's a 2D
# convolution. The final value, 32, is the number of output channels that we
# have - the number of features or feature maps that will be produced from this
# convolution. Basically, a 5 x 5 x 32 tensor will be outputted from this
# convolutional layer that will be inputted to the next one
#
# 3rd layer, we have a fully connected layer with 1024 neurons
weights = {
    'w1': tf.Variable(tf.truncated_normal([3,3,3,32], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1)),
    'w4': tf.Variable(tf.truncated_normal([2,2,64,128], stddev=0.1)),
    'w5': tf.Variable(tf.truncated_normal([2,2,128,128], stddev=0.1)),
    'w6': tf.Variable(tf.truncated_normal([8*8*128,1024], stddev=0.1)),# output of previous layer should be 7*7 image with 64 channels so flatten in final layer by multipling
    'w7': tf.Variable(tf.truncated_normal([1024,N_CLASSES], stddev=0.1))
}

# We also create our bias variable with a component for each output channel.
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[32])),
    'b2': tf.Variable(tf.constant(0.1, shape=[64])),
    'b3': tf.Variable(tf.constant(0.1, shape=[64])),
    'b4': tf.Variable(tf.constant(0.1, shape=[128])),
    'b5': tf.Variable(tf.constant(0.1, shape=[128])),
    'b6': tf.Variable(tf.constant(0.1, shape=[1024])),
    'b7': tf.Variable(tf.constant(0.1, shape=[N_CLASSES])),
}

def apply_cnn(x, weights, biases, useBatchNorm=True, useDropout=True):
    # Apply convolution to image (1st conv layer)
    h_conv1 = conv2d(x, weights['w1']) + biases['b1']
    print(h_conv1)

    # Apply ReLU to image and pool (1st Pooling layer)
    h_conv1 = tf.nn.relu(h_conv1)
    h_conv1_norm = tf.contrib.layers.batch_norm(h_conv1, is_training=is_train, center=True, scale=True) if useBatchNorm else h_conv1
    h_pool1 = tf.layers.max_pooling2d(h_conv1_norm, (3,3),3)

    # Add dropout
    h_pool1_drop = tf.nn.dropout(h_pool1, 0.9) if useDropout else h_pool1
    print(h_pool1_drop)

    # Apply convolution and relu to image (2nd conv layer and 3rd conv layer):
    # (CONV => RELU) * 2 => POOL
    h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, weights['w2']) + biases['b2'])
    h_conv2_norm = tf.contrib.layers.batch_norm(h_conv2, is_training=is_train, center=True, scale=True) if useBatchNorm else h_conv2
    print(h_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2_norm, weights['w3']) + biases['b3'])
    h_conv3_norm = tf.contrib.layers.batch_norm(h_conv3, is_training=is_train, center=True, scale=True) if useBatchNorm else h_conv3
    print(h_conv3)
    h_pool3 = tf.layers.max_pooling2d(h_conv3_norm, (2,2),2)
    print(h_pool3)

    # Add dropout
    h_pool3_drop =tf.nn.dropout(h_pool3, 0.9) if useDropout else h_pool3

    # another set of   (CONV => RELU) * 2 => POOL
    h_conv4 = tf.nn.relu(conv2d(h_pool3_drop, weights['w4']) + biases['b4'])
    h_conv4_norm = tf.contrib.layers.batch_norm(h_conv4, is_training=is_train, center=True, scale=True) if useBatchNorm else h_conv4
    print(h_conv4)
    h_conv5 = tf.nn.relu(conv2d(h_conv4_norm, weights['w5']) + biases['b5'])
    h_conv5_norm = tf.contrib.layers.batch_norm(h_conv5, is_training=is_train, center=True, scale=True) if useBatchNorm else h_conv5
    print(h_conv5)
    h_pool6 = tf.layers.max_pooling2d(h_conv5, (2,2),2)
    print(h_pool6)

    # Add dropout
    h_pool6_drop = tf.nn.dropout(h_pool6, 0.9) if useDropout else h_pool6

    # Fully connected layer
    print(h_pool6_drop)
    h_pool7_flat = tf.reshape(h_pool6_drop, [-1, 8*8*128]) # flatten
    h_fc1 = tf.nn.relu(tf.matmul(h_pool7_flat, weights['w6']) + biases['b6']) # apply weights
    h_fc1_norm = tf.contrib.layers.batch_norm(h_fc1, is_training=is_train, center=True, scale=True) if useBatchNorm else h_fc1
    h_fc1_drop = tf.nn.dropout(h_fc1_norm, 0.9) if useDropout else h_fc1_norm

    # Final, output layer
    pred = tf.matmul(h_fc1_drop, weights['w7']) + biases['b7']

    return pred


# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

pred = apply_cnn(x, weights, biases, useBatchNorm=False, useDropout=True)

pred_probs = tf.nn.softmax(pred)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y_))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred_probs,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
precision, prec_op = tf.metrics.precision(labels=tf.argmax(y_,1), predictions=tf.argmax(pred_probs,1))
recall, rec_op = tf.metrics.recall(labels=tf.argmax(y_,1), predictions=tf.argmax(pred_probs,1))
confusion_matrix = tf.confusion_matrix(labels=tf.argmax(y_,1), predictions=tf.argmax(pred_probs,1))

# Need to add local_variables_initializer because TF creates local variables
# for calculating stuff in its "metrics" package
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)

    if(isTrainingPhase):
        ds = ExhibitDataset()
        for epoch in range(EPOCHS):
            total_cost = 0
            num_batches = int(ds.train_size()/BATCH_SIZE)
            for i in range(num_batches):
                batch_x, batch_y = ds.next_batch_train(BATCH_SIZE)
                #print(sess.run(pred_probs, feed_dict={x: batch_x, y_: batch_y}))
                #print(sess.run(batch_y, feed_dict={x: batch_x, y_: batch_y}))
                c = sess.run(cost, feed_dict={x: batch_x, y_: batch_y, is_train:True})
                sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y, is_train:True})
                total_cost += c
            avg_cost = total_cost/num_batches
            print "Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost)
            if(epoch % 20 == 0):
                test_x, test_y = ds.get_test_data()
                acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y, is_train:False})
                pre = sess.run(prec_op, feed_dict={x: test_x, y_: test_y, is_train:False})
                rec = sess.run(rec_op, feed_dict={x: test_x, y_: test_y, is_train:False})
                cm = sess.run(confusion_matrix, feed_dict={x: test_x, y_: test_y, is_train:False})
                print "Test Accuracy: ", "{:.3f}".format(acc)
                print "Precision: ", "{:.3f}".format(pre)
                print "Recall: ", "{:.3f}".format(rec)
                print prettify_confusion_matrix(cm)
                # batch_x, batch_y = ds.next_batch_train(BATCH_SIZE)
                # acc = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, is_train:False})
                # print "Train Accuracy: ", "{:.5f}".format(acc)
            ds.shuffle_train_data()
        print "\nTraining complete!"

        # Save checkpoint
        save_path = saver.save(sess, "serial/PopCNN.ckpt")
        print("Model saved in file: %s" % save_path)
        # Save graph
        tf.train.write_graph(sess.graph, 'serial', 'PopCNN.pbtxt')
    elif(isContinuingTraining):
        print("-------- RESUMING TRAINING FROM SERIALIZED POINT --------")

        # Restore model weights from previously saved model
        model_path = "serial/PopCNN.ckpt"
        saver.restore(sess, model_path)

        ds = ExhibitDataset()
        for epoch in range(EPOCHS):
            total_cost = 0
            num_batches = int(ds.train_size()/BATCH_SIZE)
            for i in range(num_batches):
                batch_x, batch_y = ds.next_batch_train(BATCH_SIZE)
                c = sess.run(cost, feed_dict={x: batch_x, y_: batch_y, is_train:True})
                sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y, is_train:True})
                total_cost += c
            avg_cost = total_cost/num_batches
            print "Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost)
            if(epoch % 20 == 0):
                test_x, test_y = ds.get_test_data()
                acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y, is_train:False})
                pre = sess.run(prec_op, feed_dict={x: test_x, y_: test_y, is_train:False})
                rec = sess.run(rec_op, feed_dict={x: test_x, y_: test_y, is_train:False})
                cm = sess.run(confusion_matrix, feed_dict={x: test_x, y_: test_y, is_train:False})
                print "Test Accuracy: ", "{:.3f}".format(acc)
                print "Precision: ", "{:.3f}".format(pre)
                print "Recall: ", "{:.3f}".format(rec)
                print prettify_confusion_matrix(cm)
                # batch_x, batch_y = ds.next_batch_train(BATCH_SIZE)
                # acc = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, is_train:False})
                # print "Train Accuracy: ", "{:.5f}".format(acc)
            ds.shuffle_train_data()
        print "\nTraining complete!"

        # Save checkpoint
        save_path = saver.save(sess, "serial/PopCNN.ckpt")
        print("Model saved in file: %s" % save_path)
        # Save graph
        tf.train.write_graph(sess.graph, 'serial', 'PopCNN.pbtxt')
    elif(isTesting):
        # Restore model weights from previously saved model
        model_path = "serial/PopCNN.ckpt"
        saver.restore(sess, model_path)

        ds = ExhibitDataset()
        test_x, test_y = ds.get_test_data()
        acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y, is_train:False})
        print "Test Accuracy: ", "{:.3f}".format(acc)
    elif(not query_file_path == ''):
        # Load test image
        images = []
        img = io.imread(query_file_path)
        img_resized = transform.resize(img, IMAGE_DIMS)
        images.append(img_resized)
        # scale the raw pixel intensities to the range [0, 1]
        images = np.array(images, dtype="float") / 255.0

        # Restore model weights from previously saved model
        model_path = "serial/PopCNN2.ckpt"
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
        y_pred = sess.run(pred_probs, feed_dict={x: images, is_train:False})
        prediction = tf.squeeze(y_pred) # convert prediction to single vector as in [vocab_size] instead of [1 x vocab_size]
        print(sess.run(prediction))
        predicted_idx = sess.run(tf.argmax(prediction))
        print("{} ({}%)".format(TARGETS[predicted_idx], sess.run(prediction[predicted_idx])))

        # # Save checkpoint
        # save_path = saver.save(sess, "Serial/DexCNN.ckpt")
        # print("Model saved in file: %s" % save_path)
        # # Save graph
        # tf.train.write_graph(sess.graph, 'Serial', 'DexCNN.pbtxt')

    else:
        #ExhibitDataset()
        print("[PLEASE PROVIDE A FLAG TO SPECIFY WHICH MODE TO RUN THE "\
                "SCRIPT. SEE --help FOR MORE INFO.]")
