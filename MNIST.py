from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout	
from keras.utils import np_utils
from keras import regularizers
import matplotlib.pyplot as plt

# Seed for reproducability
np.random.seed(1671)

# Define a few parameters
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
NB_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

# Extract the datasets
(X_train,y_train), (X_test,y_test) = mnist.load_data()
RESHAPED = 28*28
X_train = X_train.reshape(60000,RESHAPED)
X_test = X_test.reshape(10000,RESHAPED)
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = np_utils.to_categorical(y_train,NB_CLASSES)
y_test = np_utils.to_categorical(y_test,NB_CLASSES)

# Normalize the pixels
X_train /= 255
X_test /= 255

# Define the model layers
model = Sequential([
	Dense(N_HIDDEN,input_shape=(RESHAPED,)),
	Activation('relu'),
	Dropout(DROPOUT),
	Dense(N_HIDDEN),
	Activation('relu'),
	Dropout(DROPOUT),
	Dense(NB_CLASSES),
	Activation('softmax'),
		])

# Model summary
model.summary()

# Configure the model for training
model.compile(loss='kullback_leibler_divergence',optimizer='adam',metrics=['accuracy'])

# Train and validate the model for a given epochs
history = model.fit(X_train,y_train,
				batch_size=BATCH_SIZE, epochs=NB_EPOCH,
				verbose = VERBOSE, validation_split=VALIDATION_SPLIT)

# Generalize the model on the test set
score = model.evaluate(X_test,y_test,verbose=VERBOSE)

# Print the accuracy
print("Test score:",score[0])
print("Test accuracy:",score[1])

# Plot the Accuracy vs epoch on the training and test data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



