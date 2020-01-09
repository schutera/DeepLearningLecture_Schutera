'''
Displaying the basic structure of a neural network and the regularization process on the mnist dataset
python -V 3.6.7
Contact: mark.schutera@kit.edu
'''

# version 1.3.0
import tensorflow as tf
# version 1.13.3
import numpy as np

np.random.seed(2)

# Define model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),

  # 0. without regularization:
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),

  # 1. L2 Parameter norm penalty by kernel regularizer:
  # tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),

  # 2. Dropout:
  # tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.5),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.5),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.5),
  # tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Define training parameters
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load training data (reduce training data to 10k samples)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_validation, y_validation) = mnist.load_data()

# Normalize input images (comply with activation function)
x_train, x_validation = x_train / 255.0, x_validation / 255.0

# 3. Augmentation
# x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
# datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                                                                 featurewise_center=True,
#                                                                 featurewise_std_normalization=True,
#                                                                 rotation_range=20,
#                                                                 width_shift_range=0.2,
#                                                                 height_shift_range=0.2,
#                                                                 horizontal_flip=False
#                                                                 )
#
# for e in range(10):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagenerator.flow(x_train, y_train, batch_size=32):
#         model.fit(np.reshape(x_batch, (-1, 28, 28)), y_batch, shuffle=True)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break

# 4. Early stopping (usually you should monitor the validation accuracy)
es = tf.keras.callbacks.EarlyStopping(monitor='acc',
                                      min_delta=0,
                                      patience=1,
                                      mode='auto'
                                      )

# Fit model on training data (with callback)
model.fit(x_train, y_train, epochs=10, shuffle=True, callbacks=[es])

# Fit model on training data (without augmentation)
# model.fit(x_train, y_train, epochs=10, shuffle=True)

# Evaluate performance on validation set
_, validation_acc = model.evaluate(x_validation, y_validation)
print('validation accuracy:', validation_acc)


# ---------------------------------------------------------------------------------------------------------------------

# 0. without regularization:
# training accuracy: 0.9919
# validation accuracy: 0.9801

# 1. L2 parameter norm penalties:
# training accuracy: 0.9450
# validation accuracy: 0.9454

# 2. Dropout 0.5:
# training accuracy: 0.9616
# validation accuracy: 0.9729

# 3. Augmentation:
# training accuracy: 0.9688
# validation accuracy: 0.9778

# 4. Early stopping:
# training accuracy: 0.9915
# validation accuracy: 0.9809

# Note: As we observe, regularization does not always yield a beneficial effect. Especially when a model already
# performs near to optimal on unobserved data (as our model does on mnist). This also correlates to the large data
# available for the task at hand.
# Also regularization hampers the training process. Try applying the regularization strategies for a larger number of
# training epochs.
