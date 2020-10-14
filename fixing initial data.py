import pandas as pd
import os
import glob
from ast import literal_eval
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def labels_to_ints(dataframe):
    d = {'A': 0, 'B': 1, 'C': 2,
         'D': 3, 'E': 4, 'F': 5,
         'G': 6}

    new_frame = [d[k] for k in dataframe]
    return new_frame

owd = os.getcwd()

os.chdir('Data/Arson/test')
arson_test_x = np.load('Arson_test.npy')

file = glob.glob("*.csv")[0]
df = pd.read_csv(file)

series = labels_to_ints(df['FIRE_SIZE_CLASS'])
arson_test_label = np.array(series)
arson_test_y = arson_test_label.reshape((arson_test_label.shape[0],1))

arson_test_y_encoded = to_categorical(arson_test_y)

os.chdir(owd)

import modeling

learning_rate = 0.01
epochs = 10
batch_size = 50
validation_split = 0.2
input_shape = arson_test_x.shape[1:4]
loss_function = sparse_categorical_crossentropy
optimizer = Adam(learning_rate =learning_rate)

#Establish the model's topography (Call your model function)
my_model = modeling.create_model(learning_rate, input_shape)

# Train model on the normalized training set.
epochs, hist = modeling.train_model(my_model, arson_test_x, arson_test_y,
                           epochs, batch_size, validation_split)




