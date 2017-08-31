#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import re
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

exclude = set(['。', '！', '？'])

exclude2 = set(['&'])

def savePath(path):
    fileList = []
    allFileNum = 0
    files = os.listdir(path)
    for f in files:  
        if(os.path.isfile(path + '/' + f)):   
            if (f != ".DS_Store"):
              fileList.append(path + '/' + f)
    return fileList, allFileNum

def read_files(fileList):
  words = []
  paragraphs = []
  for f in fileList:
    data = []
    file_object = open(f)
    try:
      all_the_text = file_object.read().decode('gb2312', 'ignore')
      all_the_text = all_the_text.encode('utf8')
      all_the_text.strip()
      all_the_text = re.sub(r'$\s*', "", all_the_text)
      all_the_text = re.sub(r'/(\w*)\s', " ", all_the_text)
      data = all_the_text.split()
      paragraphs.append(data)
      words.extend("&")
      words.extend(data)
    finally:
      file_object.close()
  return words, paragraphs

# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data


# start of data initialize of 人民日报 corpus
def initializeOfRMRB():
  filename = '/Users/xiabo/word2vec/19980105.zip'
  wordsT = read_data(filename)
  words = []
  for x in wordsT:
    matchObj = re.match( r'(.*)/(.*)', x, re.M|re.I)
    if matchObj.group(2) != 'm':
      words.append(matchObj.group(1))

  del wordsT
  sentences = []
  curSen = []

  for x in words:
    if x not in exclude:
      curSen.append(x)
    else:
      curSen.append(x)
      sentences.append(curSen)
      curSen = []
  return words, sentences
# end of data initialize of 人民日报 corpus

# start of data initialize of hsk corpus
def initializeOfHSK():
  fileList, allFileNum = savePath('/Users/xiabo/word2vec/segmented')
  wordsT, paragraphT = read_files(fileList)
  return wordsT, paragraphT
# end of data initialize of hsk corpus

# words, sentences = initializeOfRMRB()
words, sentences = initializeOfHSK()
paragraph_size = len(sentences)

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 5000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
paragraph_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
# 一次训练batch_size对参数，in skip-gram模型
# PV concatenated WV to predict next word
# Shareing PV not across paragraph
######################################################

def generate_batch(batch_size, num_skips, skip_window, corpus):
  global data_index
  global paragraph_index
  global exclude
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  para_input = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
    # print(data_index, reverse_dictionary[data[data_index]],'->', paragraph_index)

  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)

      batch[i * num_skips + j] = buffer[skip_window]
      para_input[i * num_skips + j] = paragraph_index
      labels[i * num_skips + j, 0] = buffer[target]

      if corpus == 'rmrb':
        if reverse_dictionary[buffer[target]] in exclude:
          paragraph_index = (paragraph_index + 1) % paragraph_size
      if corpus == 'hsk':
        if reverse_dictionary[buffer[target]] in exclude2:
          paragraph_index = (paragraph_index + 1) % paragraph_size
        
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
    
  return batch, labels, para_input

# batch, labels, para_labels = generate_batch(batch_size=128, num_skips=2, skip_window=2, corpus="hsk")
# print(batch)
# print(labels)
# print(para_labels)

# for i in range(128):
#   print(batch[i], reverse_dictionary[batch[i]],
#         '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
# exit()
# 257 希望 -> 1197 充满
# 257 希望 -> 3483 迈向
# batch: inputs; 
# labels: target classes;

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 5     # Random set of words to evaluate similarity on.
valid_window = 10  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_word_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_para_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_word_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    word_embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    para_embeddings = tf.Variable(
        tf.random_uniform([paragraph_size, embedding_size], -1.0, 1.0))
    #tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

    word_embed = tf.nn.embedding_lookup(word_embeddings, train_word_inputs)
    para_embed = tf.nn.embedding_lookup(para_embeddings, train_para_inputs)

    embed = (word_embed + para_embed) / 2
    # embed = (word_embed + para_embed)
    #embed = tf.reduce_sum(tf.concat(1, [word_embed, para_embed]), 1)
    # embed = tf.div(tf.reduce_sum(tf.concat(1, [word_embed, para_embed]), 1), skip_window*2 + 1)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.

  ######################################################
  # Alter loss function to adapt PV + WV

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_word_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity of word_embeddings between minibatch examples and all embeddings.
  word_norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
  word_normalized_embeddings = word_embeddings / word_norm
  word_valid_embeddings = tf.nn.embedding_lookup(
      word_normalized_embeddings, valid_dataset)
  word_similarity = tf.matmul(
      word_valid_embeddings, word_normalized_embeddings, transpose_b=True)

  # Compute the cosine similarity of para_embeddings between minibatch examples and all embeddings.
  para_norm = tf.sqrt(tf.reduce_sum(tf.square(para_embeddings), 1, keep_dims=True))
  para_normalized_embeddings = para_embeddings / para_norm
  para_valid_embeddings = tf.nn.embedding_lookup(
      para_normalized_embeddings, [10])
  para_similarity = tf.matmul(
      para_valid_embeddings, para_normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels, paragraph_inputs = generate_batch(
        batch_size, num_skips, skip_window, 'hsk')
    # return batch, labels, para_input

    feed_dict = {train_word_inputs: batch_inputs, train_para_inputs: paragraph_inputs, train_word_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    # word embedding similarity represitation
    if step % 10001 == 0:
      sim = word_similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          #close_word = reverse_dictionary[(nearest[k]%len(reverse_dictionary))]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)

    # paragraph embedding similarity represitation
      # print(valid_examples)

      sim = para_similarity.eval()
      # for i in xrange(valid_size):
      #   print("Nearest sentences:")
      #   for j in sentences[valid_examples[0]]:
      #     print(j, end='')

      #   top_k = 8  # number of nearest neighbors
      #   nearest = (-sim[0, :]).argsort()[1:top_k + 1]
      #   print("\n")
      #   for k in xrange(top_k):
      #     for f in sentences[nearest[k]]:
      #       print(f, end='')
      #     print("\n")
      
      print("Nearest sentences:")
      # for j in sentences[820]:
      #   print(j, end='')

      top_k = 9  # number of nearest neighbors
      nearest = (-sim[0, :]).argsort()[1:top_k + 1]
      print("\n")
      for k in xrange(top_k):
        for f in sentences[nearest[k]]:
          print(f, end='')
        print("\n")

  # final_embeddings_word = word_normalized_embeddings.eval()
  # final_embeddings_para = para_normalized_embeddings.eval()
