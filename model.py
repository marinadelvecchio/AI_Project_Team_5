# -*- coding: utf-8 -*-
"""
Created on Mon May  8 03:13:08 2023

@author: tyele
"""

#%% Setup
import pandas as pd
import keras
import numpy as np
import os

gpus = keras.backend._get_available_gpus()

if len(gpus) == 0:
    print("No GPU's Available")
else:
    print(gpus)


current_path = "E:/Github/AI_Project_Team_5/"


#%% Functions

from keras.preprocessing import *
from keras.utils import pad_sequences

import math
import re
import numpy as np




genre_dict = {
"Comedy": 0,
"Crime": 1,
"Drama": 2,
"Action": 3,
"Horror": 4,
"Thriller": 5,
"Mystery": 6,
"War": 7,
"Fantasy": 8,
"Foreign": 9,
"Music": 10,
"Documentary": 11,
"Romance": 12,
"Adventure": 13,
"Science Fiction": 14,
"Family": 15,         
"Western": 16,
"TV Movie": 17,
"Animation": 18,
"History": 19
} 

col = {'userID': 0,
       'genre': 1,
       'overview': 2,
       'budget': 3,
       'rating': 4,
       'collection': 5}

def get_user_movies(df, ID):
    return df[df['userId'] == ID]

def df_to_numpy(df):
    df_np = np.array([
        df['userId'],
        df['genres'],
        df['overview'],
        df['budget'],
        df['rating'],
        df['belongs_to_collection'],
        df['title'],
        df['like_movie']]
    )
    return df_np
def pattern_to_list(data, pattern):
    l = list()
    for i in range(len(data)):
        l.append(re.findall(pattern, data[i,]))
    return l 

def one_hot_encode(data, dictionary):
    nrow = len(data)
    ncol = len(dictionary)
    
    
    matrix = np.zeros(shape = (nrow,ncol))
    for i in range(nrow):
        for k in range(len(data[i])):
            col = dictionary[data[i][k]]
            matrix[i,col] = 1.0
    return matrix
def datasplit(data, ratio = .8):
    data = data.sample(frac = 1)
    
    train_end = math.floor(len(data) * .8)
    train = data[0:train_end]
    test = data[train_end:len(data)]
    
    return train, test
    


def generate_sequences(text_data,#training data
                      maxlen = 50,#maximum length of the embedding sequence
                      max_words = 2000,
                      tokenizer = None):#will only choose consider max_words amount of words for the embedding
    if tokenizer == None:
        tok = text.Tokenizer(num_words=max_words)#Create tokenizer
        tok.fit_on_texts(text_data)#fit tokenizer on texts
    else:
        tok = tokenizer
    
    sequences = tok.texts_to_sequences(text_data) #generate sequences from tokenizer
    
    padded_sequences = pad_sequences(sequences, maxlen = maxlen)#pad sequneces to same length
    
    return (padded_sequences, tok)#return the sequences and the tokenizer

#def kfold(data, model, nfolds = 10):
#    data
    
    
def preprocess(df, maxlen = 50, max_words = 100, dict_tokenizer = None):
    df_np = df_to_numpy(df)
    genre_x = df_np[1,:]#Done
    overview_x = df_np[2,:]#Done
    budget_x = df_np[3,:]#Done
    title_x = df_np[6,:]
    rating_x = df_np[4,:]
    
    y_like_movie = df_np[7,:].astype('int')

    pattern = r"'name': '([^']*)'"
    
    
    genre_list = pattern_to_list(genre_x, pattern)
    
    
    if dict_tokenizer != None:
       overview_tokenizer = dict_tokenizer['overview_tokenizer']
       title_tokenizer = dict_tokenizer['title_tokenizer']
    else:
       overview_tokenizer = None
       title_tokenizer = None
    
    
    x_genre = one_hot_encode(genre_list,genre_dict)
    x_overview, overview_tokenizer = generate_sequences(overview_x, maxlen = maxlen, max_words = max_words, tokenizer=overview_tokenizer)
    x_title, title_tokenizer = generate_sequences(title_x, maxlen = maxlen, max_words = max_words, tokenizer=title_tokenizer)
    x_budget = np.asarray(budget_x).astype('float32')
    
    return x_genre,x_overview,overview_tokenizer, x_title, title_tokenizer, x_budget, y_like_movie


def kfold_lstm(data, nfolds = 10):
    
    data = data.sample(frac = 1) #Shuffle
    accuracy = list()
    
    num_obs = len(data)
    
    index_increment = math.floor(num_obs/10)
    
    
    start_index = 0
    
    num_obs = len(data)
    
    for i in range(nfolds):
        
        
        print("Fold: " + str(i+1))
        
        
        
        test_range = range(start_index, start_index + index_increment)
        start_index = start_index + index_increment
       
        
        if(test_range.start == 0):
            train = data[test_range.stop:]
            print("Train Range: " + str(test_range.stop) + " ---> " + str(num_obs))
            
            
            
            
        elif(test_range.stop == num_obs):
            train = data[:test_range.start]
            print("Train Range: " + "0 ---> " + str(test_range.start))
            
            
        else:
            train = data[:test_range.start] + data[test_range.stop:]
            print("Train Range: " + "0 ---> " + str(test_range.start) + " , " + str(test_range.stop) + " ---> " + str(num_obs))
        print("Test Range: " + str(test_range.start) + " ---> " + str(test_range.stop))
        print("*_____________________________________________________________________*")
        
        test = data[test_range.start:test_range.stop]
    
        x_genre,x_overview,overview_tokenizer, x_title, title_tokenizer, x_budget,y_like_movie = preprocess(train)
        t_x_genre,t_x_overview,t_overview_tokenizer, t_x_title, t_title_tokenizer, t_x_budget,t_y_like_movie = preprocess(test, dict_tokenizer={'overview_tokenizer':overview_tokenizer,
                                                                                                                                  'title_tokenizer':title_tokenizer})
    
        

        o_x = tf.keras.Input(shape = (maxlen,)) #Overview
        o_x_2 = layers.Embedding(input_dim = max_words,
                               input_length = maxlen,
                               output_dim = 8)(o_x)
        #o_x_3 = layers.Reshape((maxlen, 8))(o_x_2)
        o_x_3 = layers.LSTM(64)(o_x_2)

        g_x = tf.keras.Input(shape = len(x_genre[0,])) #Genre

        b_x = layers.Input(shape = 1) #Budget
        #r_x = layers.Input(shape = ()) #Rating


        t_x = tf.keras.Input(shape = (maxlen,)) #Title
        t_x_2 = layers.Embedding(input_dim = max_words,
                               input_length = maxlen,
                                 output_dim = 8)(t_x)
        #t_x_3 = layers.Reshape((maxlen, 8))(t_x_2)
        t_x_3 = layers.LSTM(64)(t_x_2)




        concat_layers = layers.Concatenate()([o_x_3,g_x, b_x,t_x_3])
        dense1 = keras.layers.Dense(2, use_bias=True)(concat_layers)
        output = keras.layers.Dense(1, activation=keras.activations.sigmoid, use_bias=True)(dense1)
        #z = layers.Dense(2, activaion="relu")(concat_layers)






        model = keras.Model(inputs=[o_x,
                               g_x,
                               b_x,
                               t_x],
                            outputs=output
                           )

        model.compile(
            optimizer = keras.optimizers.RMSprop(lr=0.01),
            loss = 'binary_crossentropy',
            metrics=['accuracy']
        )




        history = model.fit([x_overview,
                   x_genre,
                   x_budget,
                   x_title
                  ],
                  y_like_movie,
                 batch_size = 16,
                 epochs = 5,
                 verbose=1)
        
        Predictions = model.predict([t_x_overview,
                   t_x_genre,
                   t_x_budget,
                   t_x_title
                  ],
                  t_y_like_movie)
        
        print(Predictions)
        
        
        
        
        
        
    
#%% LSTM Model


import tensorflow as tf
#from tensorflow import keras
from tensorflow import keras
from keras import optimizers



from tensorflow.keras import layers



maxlen = 50
max_words = 200


dataset_dir = current_path + "Datasets/FinalClean.csv"
df = pd.read_csv(dataset_dir)
df = df.iloc[:,1:21]
df_564 = get_user_movies(df, 564)


kfold_lstm(df_564, nfolds = 10)


#%%%



train, test = datasplit(df_564)

x_genre,x_overview,overview_tokenizer, x_title, title_tokenizer, x_budget y_like_movie = preprocess(train)


#T_belongs_to_collection_x = df_np[5,:]


num_obs = 564

o_x = tf.keras.Input(shape = (maxlen,)) #Overview
o_x_2 = layers.Embedding(input_dim = max_words,
                       input_length = maxlen,
                       output_dim = 8)(o_x)
#o_x_3 = layers.Reshape((maxlen, 8))(o_x_2)
o_x_3 = layers.LSTM(64)(o_x_2)

g_x = tf.keras.Input(shape = len(x_genre[0,])) #Genre

b_x = layers.Input(shape = 1) #Budget
#r_x = layers.Input(shape = ()) #Rating


t_x = tf.keras.Input(shape = (maxlen,)) #Title
t_x_2 = layers.Embedding(input_dim = max_words,
                       input_length = maxlen,
                         output_dim = 8)(t_x)
#t_x_3 = layers.Reshape((maxlen, 8))(t_x_2)
t_x_3 = layers.LSTM(64)(t_x_2)




concat_layers = layers.Concatenate()([o_x_3,g_x, b_x,t_x_3])
dense1 = keras.layers.Dense(2, use_bias=True)(concat_layers)
output = keras.layers.Dense(1, activation=keras.activations.sigmoid, use_bias=True)(dense1)
#z = layers.Dense(2, activaion="relu")(concat_layers)






model = keras.Model(inputs=[o_x,
                       g_x,
                       b_x,
                       t_x],
                    outputs=output
                   )

model.compile(
    optimizer = keras.optimizers.RMSprop(lr=0.01),
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)




history = model.fit([x_overview,
           x_genre,
           x_budget,
           x_title
          ],
          y_like_movie,
         batch_size = 16,
         epochs = 20,
         verbose=1)



#%%% Create Predictions

model.predict()





#%% Visualzie Results


from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])
plt.show()


