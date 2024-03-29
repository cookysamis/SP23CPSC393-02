# -*- coding: utf-8 -*-
"""S_Kim_HW4_SP23.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q0EwG30UtX52th5SZAmotOJf7Lz5kBo_

# Homework 4 (Sequential Models)

1. Choose a book or other text to train your model1 on (I suggest [Project Gutenberg](https://www.gutenberg.org/ebooks/) to find .txt files but you can find them elsewhere). Make sure your file is a `.txt` file. Clean your data (You may use [this file](https://colab.research.google.com/drive/1HCgKn5XQ7Q3ywxGszVWx2kddfT9UBASp?usp=sharing) we talked about in class as a baseline). Build a sequential model1 (GRU, GRU, SimpleRNN, or Transformer) that generates new lines based on the training data (NO TRANSFER LEARNING).

Print out or write 10 generated sequences from your model1 (Similar to Classwork 17 where we generated new Pride and Prejudice lines, but now with words instead of charachters. Feel free to use [this](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model1-in-keras/) as a reference). Assess in detail how good they are, what they're good at, what they struggle to do well. 

2. Make a new model1 with ONE substantial adjustment (e.g. use a custom embedding layer if you didn't already, use a pre-trained embedding layer if you didn't already, use a DEEP GRU/GRU with multiple recurrent layers, use a pre-trained model1 to do transfer learning and fine-tune it...etc.). 

Print out or write 10 generated sequences from your model1 (Similar to Classwork 17 where we generated new Pride and Prejudice lines, but now with words instead of charachters. Feel free to use [this](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model1-in-keras/) as a reference). Assess in detail how good they are, what they're good at, what they struggle to do well.  Did the performance of your model1 change?

3. Then create a **technical report** discussing your model1 building process, the results, and your reflection on it. The report should follow the format in the example including an Introduction, Analysis, Methods, Results, and Reflection section.

# Introduction
An introduction should introduce the problem you're working on, give some background and relevant detail for the reader, and explain why it is important. 

# Analysis 
Any exploratory analysis of your data, and general summarization of the data (e.g. summary statistics, correlation heatmaps, graphs, information about the data...). This can also include any cleaning and joining you did. 

# Methods
Explain the structure of your model1 and your approach to building it. This can also include changes you made to your model1 in the process of building it. Someone should be able to read your methods section and *generally* be able to tell exactly what architechture you used. 

# Results
Detailed discussion of how your model1 performed, and your discussion of how your model1 performed.

# Reflection
Reflections on what you learned/discovered in the process of doing the assignment. Things you would do differently in the future, ways you'll approach similar problems in the future, etc.
"""

# modified from: https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model1-in-keras/
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from pickle import dump, load
from random import randint

import string

# changeable params
my_file = "mobydick.txt"
seq_len = 100

# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text
 
# turn a doc into clean tokens
def clean_doc(doc):
 # replace '--' with a space ' '
 doc = doc.replace('--', ' ')
 # split into tokens by white space
 tokens = doc.split()
 # remove punctuation from each token
 table = str.maketrans('', '', string.punctuation)
 tokens = [w.translate(table) for w in tokens]
 # remove remaining tokens that are not alphabetic
 tokens = [word for word in tokens if word.isalpha()]
 # make lower case
 tokens = [word.lower() for word in tokens]
 return tokens
 
# save tokens to file, one dialog per line
def save_doc(lines, filename):
 data = '\n'.join(lines)
 file = open(filename, 'w')
 file.write(data)
 file.close()
 
# load document
doc = load_doc(my_file)
print(doc[:200])
 
# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))
 
# organize into sequences of tokens
length = seq_len + 1
sequences = list()
for i in range(length, len(tokens)):
 # select sequence of tokens
 seq = tokens[i-length:i]
 # convert into a line
 line = ' '.join(seq)
 # store
 sequences.append(line)
print('Total Sequences: %d' % len(sequences))
 
# save sequences to file
out_filename = my_file[:-4] + '_seq.txt'
save_doc(sequences, out_filename)

# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text
 
# load
doc = load_doc(out_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(lines)
sequences = tokenizer1.texts_to_sequences(lines)

# vocabulary size
vocab_size = len(tokenizer1.word_index) + 1


# separate into input and output
sequences = np.array(sequences)
sequences.shape
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

p_train = 0.8

n_train = int(X.shape[0]//(1/p_train))
X_train = X[0:n_train]
y_train = y[0:n_train]
X_test = X[n_train:]
y_test = y[n_train:]

X_train.shape

X_test.shape

y_train

y_test

y_train.shape

y_test.shape

# define model1
model1 = Sequential()
model1.add(Embedding(vocab_size, 50, input_length=seq_length))
model1.add(GRU(128, return_sequences=True))
model1.add(GRU(64))
model1.add(Dense(100, activation='relu'))
model1.add(Dense(vocab_size, activation='softmax'))
print(model1.summary())
# compile model1
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model1
model1.fit(X, y, batch_size=128, epochs=100)

# save the model1 to file
model1.save('model1.h5')
# save the tokenizer1
dump(tokenizer1, open('tokenizer1.pkl', 'wb'))

# generate a sequence from a language model1
def generate_seq(model1, tokenizer1, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer1.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		predict_y = model1.predict(encoded)
		yhat = np.argmax(predict_y, axis = 1)
		# yhat = model1.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer1.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

# load cleaned text sequences
in_filename = 'mobydick_seq.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model1
model1 = load_model('model1.h5')

# load the tokenizer1
tokenizer1 = load(open('tokenizer1.pkl', 'rb'))

for i in range (10):
	# select a seed text
	seed_text = lines[randint(0,len(lines))]
	print(seed_text + '\n')

	# generate new text
	generated = generate_seq(model1, tokenizer1, seq_length, seed_text, 50)
	print(generated)