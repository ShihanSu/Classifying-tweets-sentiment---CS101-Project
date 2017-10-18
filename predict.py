#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers_pre
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
 #   print("{}={}".format(attr.upper(), value))
#print("")

# Data Loading
#====================================================
# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw = data_helpers_pre.load_data_and_labels('test.csv')
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

#print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "checkpoints")
checkpoint_file = os.path.join(checkpoint_path, 'model-21000')
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

        # Generate batches for one epoch
        #batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        #all_predictions = []

        #for x_test_batch in batches:
          #  batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
           # all_predictions = np.concatenate([all_predictions, batch_predictions])
	
        prediction = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
        print('prediction is ', prediction)
       ## CHANGE THIS, write code to write the prediction back to database	
