'''
Let's say we have raw data like a string. We can categorize the string
using 1's and 0's. If a string is a representative of a category of interest,
we can represent it with a 1. If not, 0. We can use a dictionary and map
it, obviously.

Another thing to mention is feature crosses. This is when we multiply
two featuree sets together, and can be an incredibly useful tool for
predicting things that might be more difficult to predict with individual
features. An example could be lattitude and longitude. These could be
ok features on their own, but together they are powerful for prediction. A
neighborhood can be a great indicator of housing precises, in contrast to
just latitude and longitude.
'''

#@title Load the imports

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

print("Imported the modules.")

# Load the dataset
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Scale the labels
scale_factor = 1000.0
# Scale the training set's label.
train_df["median_house_value"] /= scale_factor

# Scale the test set's label
test_df["median_house_value"] /= scale_factor

# Shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))


'''
A feature column represents an individual feature. They are the middleware
that bridge (in this case columns from a CSV) to features used to train the model.

If we were to represent a certain feature as a floating point value, we could
use tf.feature_column.numeric_column

If we were to represent a certain feature as a series of buckets or bins, we 
would use tf.feature_column.bucketized_column

When used, the layer processes the raw inputs, according to the transformations described 
by the feature columns, and packs the result into a numeric array. 
(The model will train on this numeric array.)
'''

resolution_in_degrees = 0.4

# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                     int(max(train_df['latitude'])),
                                     resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                               latitude_boundaries)
feature_columns.append(latitude)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                      int(max(train_df['longitude'])),
                                      resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)
feature_columns.append(longitude)

# Create a feature cross of latitude and longitude.
# will be 100 bins total because 10^2, so hash bucket size is that size I'd assume
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Convert the list of feature columns into a layer that will ultimately become
# part of the model. Understanding layers is not important right now.
feature_cross_feature_layer = layers.DenseFeatures(feature_columns)


# @title Define functions to create and train a model, and a plotting function
def create_model(my_learning_rate, feature_layer):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """Feed a dataset into the model in order to train it."""

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.94, rmse.max() * 1.05])
    plt.show()


print("Defined the create_model, train_model, and plot_the_loss_curve functions.")

# The following variables are the hyperparameters.
learning_rate = 0.05
epochs = 30
batch_size = 100
label_name = 'median_house_value'

# Create and compile the model's topography.
my_model = create_model(learning_rate, feature_cross_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
test_features = {name: np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

# floating point values for latitude and longitude don't really matter too much.
# There won't be a direct correlation between them. What could be better is by
# thinking of the data differently. Thinking of it as neighborhoods, which is why
# it is better to bin them instead. Neighborhoods will have much more predictive
# power than just simple latitude and longitude as features. Be smarter with your features.

# Even bucketing these features helps a bit more, but there's still no predictive power.
# We should consider feature crossing these. Once again, be smart with your data.
# Bucketing then feature crossing those buckets is a phenomenal way to generate predictive
# power.
