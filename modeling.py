#import required packages for project
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
# import tensorflow specifics
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1


#
#Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()




def create_model(my_learning_rate, input_shape):
    """Create and compile a deep neural net."""

    # Create Sequential model
    model = tf.keras.models.Sequential()
    model = Sequential()
    # Attempting to flatten 3D array
    model.add(Conv2D(60, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(Conv2D(60, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(Conv2D(120, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(Conv2D(120, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', activity_regularizer=l1(0.001), kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(Dense(7, activation='softmax'))

    # Construct the layers into a model that TensorFlow can execute.
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.
    model.compile(optimizer=Adam(lr=my_learning_rate),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    """Train the model by feeding it data."""

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split)

    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()