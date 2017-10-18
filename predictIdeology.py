#! /usr/bin/env python

import tensorflow as tf
import os
import time
import datetime
import data_helpers_pre
from text_cnn import TextCNN
from tensorflow.contrib import learn
import numpy as np
import re
import itertools
from collections import Counter

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./ideology/runs/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "checkpoints")
checkpoint_file = os.path.join(checkpoint_path, 'model-21000')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_set(tweets):
	return [clean_str(tweet) for tweet in tweets]

def predict(raw_tweets):
	clean_tweets = clean_set(raw_tweets)
	tweets = np.array(list(vocab_processor.transform(clean_tweets)))
	graph = tf.Graph()
	with graph.as_default():
	    session_conf = tf.ConfigProto(
	      allow_soft_placement=FLAGS.allow_soft_placement,
	      log_device_placement=FLAGS.log_device_placement)
	    sess = tf.Session(config=session_conf)
	    with sess.as_default():
	        # Load the saved meta graph and restore variables
	        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
	        saver.restore(sess, checkpoint_file)

	        # Get the placeholders from the graph by name
	        input_x = graph.get_operation_by_name("input_x").outputs[0]
	        # input_y = graph.get_operation_by_name("input_y").outputs[0]
	        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

	        # Tensors we want to evaluate
	        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

	        prediction = sess.run(predictions, {input_x: tweets, dropout_keep_prob: 1.0})
	        return prediction


if __name__ == '__main__':
	text = ['republican', 'what is your name?', 'bitch who you talking to', 'fuck this im going home']
	print(predict(text))
	print(predict(text))