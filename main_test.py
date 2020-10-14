import modeling
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def labels_to_ints(dataframe):
    d = {'A': 0, 'B': 1, 'C': 2,
         'D': 3, 'E': 4, 'F': 5,
         'G': 6}

    new_frame = [d[k] for k in dataframe]
    return new_frame





def main_function():
    train_x =  np.load('Data/Lightning/test/Lightning_test.npy')
    train_y = pd.read_csv('Data/Lightning/test/Lightning_2014_2015_test.csv')
    train_y = labels_to_ints(train_y['FIRE_SIZE_CLASS'])
    train_y = np.array(train_y)

    input_shape = (train_x.shape[1], train_x.shape[2], train_x.shape[3])

    # Import ML model (must assign Learning Rate first)
    learning_rate = 0.05
    my_model = modeling.create_model(learning_rate, input_shape)

    # Assign ML model hyperparameters
    #learning_rate = 0.1
    epochs = 10
    batch_size = 100
    validation_split = 0.25


# Train model
    epochs, hist = modeling.train_model(my_model, train_x, train_y, epochs, batch_size, validation_split)
# plot
    modeling.plot_curve(epochs, hist, 'accuracy')
    print('Test modeling done')

if __name__ == '__main__':
    main_function()

