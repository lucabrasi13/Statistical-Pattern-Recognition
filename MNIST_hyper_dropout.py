from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout	
from keras.utils import np_utils
from keras import regularizers
import matplotlib.pyplot as plt

def test_and_train_run(DROPOUT):
	# Seed for reproducability
	np.random.seed(1671)

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

	# Configure the model for training
	model.compile(loss='kullback_leibler_divergence',optimizer='adam',metrics=['accuracy'])

	# Train and validate the model for a given epochs
	history = model.fit(X_train,y_train,
					batch_size=BATCH_SIZE, epochs=NB_EPOCH,
					verbose = VERBOSE, validation_split=VALIDATION_SPLIT)

	# Generalize the model on the test set
	score = model.evaluate(X_test,y_test,verbose=VERBOSE)
	
	# Print the test score accuracy
	print("Test score:",score[0])
	print("Test accuracy:",score[1])

	return score

# Global definition of a few parameters
global NB_EPOCH
global BATCH_SIZE
global VERBOSE
global NB_CLASSES
global N_HIDDEN
global NB_HIDDEN
global VALIDATION_SPLIT

# Global declaration of a few parameters
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
NB_HIDDEN = 128
VALIDATION_SPLIT = 0.2

# Define the hyperparameter to optimize
DROPOUT = np.linspace(0,0.5,11)

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

# Hyperparameter tuning
M = len(DROPOUT)
val_score = np.zeros((M,1))
val_accuracy = np.zeros((M,1))

for i in range(0,M):
	print("------------------------------------------------------------------")
	print("DROPOUT:",DROPOUT[i])
	val_score[i],val_accuracy[i] = test_and_train_run(DROPOUT[i])
	print("------------------------------------------------------------------")

# View the Test accuracy to different hyperparameter values
plt.plot(DROPOUT,val_accuracy)
plt.title('DROPOUT tuning')
plt.ylabel('Accuracy -->')
plt.xlabel('epoch -->')
plt.legend(['accuracy'], loc='upper left')
plt.show()

