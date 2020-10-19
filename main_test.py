import modeling
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import random



def labels_to_ints(dataframe):
    d = {'A': 0, 'B': 1, 'C': 2,
         'D': 3, 'E': 4, 'F': 5,
         'G': 6}

    new_frame = [d[k] for k in dataframe]
    return new_frame

def random_checker(label, ):


def sample_indexer(label_data):
    labels = list(set(label_data))
    length = len(label_data)
    ratio = int(length/len(labels))
    final_index=[]
    for label in labels:
        new_indexing = []
        i=0
        while i<ratio:
            index = random.randint(0, length-1)
            if label_data[index]==label:
                if index in new_indexing:
                    i=i
                else:
                    new_indexing.append(index)
                    i=i+1
            else:
                i=i
        final_index.append(new_indexing)

    return final_index






def main_function():


    if tf.test.gpu_device_name():

        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

    else:

        print("Please install GPU version of TF")

    type = 'Miscellaneous'
    train_x = np.load('Data/'+str(type)+'/train/'+str(type)+'_train.npy')
    train_x = (train_x/255.0)
    train_y = pd.read_csv('Data/'+str(type)+'/train/'+str(type)+'_2014_2015_train.csv')
    train_y = labels_to_ints(train_y['FIRE_SIZE_CLASS'])
    train_y_array = np.array(train_y)
    train_y_array = tf.keras.utils.to_categorical(train_y)

    input_shape = (train_x.shape[1], train_x.shape[2], train_x.shape[3])
    output_shape = train_y_array.shape[1]
    # Import ML model (must assign Learning Rate first)
    learning_rate = 0.05
    my_model = modeling.create_model(learning_rate, input_shape, output_shape)

    # Assign ML model hyperparameters
    #learning_rate = 0.1
    epochs = 10
    batch_size = 100
    validation_split = 0.3


# Train model
    epochs, hist = modeling.train_model(my_model, train_x, train_y_array, epochs, batch_size, validation_split)
# plot
    modeling.plot_curve(epochs, hist, 'accuracy')
    print('Test modeling done')

if __name__ == '__main__':
    main_function()

