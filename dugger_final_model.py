#%%
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import random
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import NumpyArrayIterator
from matplotlib import pyplot as plt

import datetime

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Code Flow:
    The code consists of five cells.  The first cell includes the library
    imports as well as the data load section.  The next five cells contain
    a model for each cell.  Models 1, 3, 4, and 5 have a Model Section and 
    Output Section.  Model 2 has an additional section called Pretreat Data.
    The cells can be run independently by clicking within the cell and
    pressing Shift + Enter.  Cell 1 must be run first to import the necessary
    libraries and load the data.  The other cells can be run as desired.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Variables:
    train_df: Pandas dataframe containing training data set
    test_df: Pandas dataframe containing test data set
    train_data: Numpy array converting all training data types to float 32
    test_data: Numpy array converting all test data types to float 32
    x_train: Numpy array containing training independent variables
    y_train: Numpy array containing training dependent variables
    x_test: Numpy array containing test independent variables
    y_test: Numpy array containing test dependent variables
    model1: keras object containing Model 1 (this variable is repeated for
                                             subsequent models with respective
                                             numbers)
    x_train_gen: keras object containing parameters for random image augmentation
    train_aug: keras object containing augmented input images
    
The following variables are reset for each model:
    start_time: variable containing datetime that model starts
    estimator: keras object containing trained model
    history_dict: dictionary containing model statistics
    loss_values: list containing training loss values
    val_loss_values: list containing test loss values
    epochs: variable containing number of epochs used in model
    acc: list containing training accuracy values
    val_acc: list containing test accuracy values
    stop_time: variable containing datetime that model stops
        
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
#read in data
train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

#display first five rows of data
train_df.head()
test_df.head()

#convert data to numpy array with data types float 32
train_data = np.array(train_df, dtype="float32")
test_data = np.array(test_df, dtype="float32")

#slice data into independent and dependent variables, divide by 255 to scale pixels
x_train = train_data[:,1:] /255
y_train = train_data[:,0]
x_test = test_data[:,1:] /255
y_test = test_data[:,0]

#reshape rows into 28x28 image shape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#encode dependent variables as categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
"""
datagen = ImageDataGenerator()
train = datagen.flow_from_directory("Alzheimers_Dataset/train",class_mode='categorical')
test = datagen.flow_from_directory("Alzheimers_Dataset/test",class_mode='categorical')

imgs_train, lables_train = next(train)
imgs_test, labels_test = next(train)

#%%
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
*******************************Model 1***********************************
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model Section M1
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#start timer
start_time = datetime.datetime.now()

#create feed forward model
model1 = models.Sequential()

random.seed(1)
#Convolution layer with filter size 32, kernel size 3, and input size of 28
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
#Pooling layer with pool_size 2
model1.add(layers.MaxPooling2D((2, 2)))
#Convolution layer with filter size 64, kernel size 3, and padding so input matches output
model1.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "same"))
#Pooling layer with pool size 2
model1.add(layers.MaxPooling2D((2, 2)))
#Convolution layer with filter size 128, kernel size 3, and padding so input matches output
model1.add(layers.Conv2D(128, (3, 3), activation='relu', padding = "same"))
#Pooling layer with pool size 2
model1.add(layers.MaxPooling2D((2, 2)))
#Flattening layer to convert to 1D
model1.add(layers.Flatten())
#Dense layer to connect to neural network
model1.add(layers.Dense(128, activation='relu'))
#Output layer to classify
model1.add(layers.Dense(4, activation='softmax'))
#Compile model
model1.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#Print summary of model
model1.summary()

#Fit model
estimator = model1.fit(train, epochs = 20, verbose = 1, validation_data = test)


#stop timer
stop_time = datetime.datetime.now()
#print elapsed time
print ("Time required for training:",stop_time - start_time)
 #%%
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Output Section M1
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#assign model values to dictionary
history_dict = estimator.history
#assign training loss values
loss_values = history_dict["loss"]
#assign test loss values
val_loss_values = history_dict["val_loss"]
#calculate epochs
epochs = range(1, len(loss_values) + 1)
#plot training loss
plt.plot(epochs, loss_values, "bo", label="Training loss")
#plot test loss
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
#title plot
plt.title("Training and validation loss M1")
#label x axis
plt.xlabel("Epochs")
#label y axis
plt.ylabel("Loss")
#show legend
plt.legend()
#save figure
plt.savefig('M1_loss.png')
#show plot
plt.show()

#clear plot
plt.clf()
#assign training accuracy values
acc = history_dict["accuracy"]
#assign test accuracy values
val_acc = history_dict["val_accuracy"]
#plot training accuracy values
plt.plot(epochs, acc, "bo", label="Training acc")
#plot test accuracy values
plt.plot(epochs, val_acc, "b", label="Validation acc")
#title plot
plt.title("Training and validation accuracy M1")
#label x axis
plt.xlabel("Epochs")
#label y axis
plt.ylabel("Accuracy")
#show legend
plt.legend()
#save figure
plt.savefig('M1_acc.png')
#show plot
plt.show()


#plot heatmap
labels = ["Mild","Mod","Non","V_Mild"]
y_pred = model1.predict(test)
cf_matrix = tf.math.confusion_matrix(test.labels, tf.argmax(y_pred,1))
sns.heatmap(cf_matrix, annot=True, cmap="Blues",xticklabels=labels, yticklabels=labels)
plt.title("M1 HeatMap")
plt.savefig('M1_heat.png')
plt.show()
#%%
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
*******************************Model 2***********************************
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model Section M2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#start timer
start_time = datetime.datetime.now()

#create feed forward model
model2 = models.Sequential()

random.seed(1)
#Convolution layer with filter size 32, kernel size 3, and input size of 28
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3), 
                         kernel_regularizer=regularizers.l2(l=0.01)))
#Pooling layer with pool_size 2
model2.add(layers.MaxPooling2D((2, 2)))
#Convolution layer with filter size 64, kernel size 3, and padding so input matches output
model2.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "same",
                         kernel_regularizer=regularizers.l2(l=0.01)))
#Pooling layer with pool size 2
model2.add(layers.MaxPooling2D((2, 2)))
#Convolution layer with filter size 128, kernel size 3, and padding so input matches output
model2.add(layers.Conv2D(128, (3, 3), activation='relu', padding = "same",
                         kernel_regularizer=regularizers.l2(l=0.01)))
#Pooling layer with pool size 2
model2.add(layers.MaxPooling2D((2, 2)))
#Flattening layer to convert to 1D
model2.add(layers.Flatten())
#Dense layer to connect to neural network
model2.add(layers.Dense(128, activation='relu'))
#Output layer to classify
model2.add(layers.Dense(4, activation='softmax'))
#Compile model
model2.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#Print summary of model
model2.summary()

#Fit model
estimator = model2.fit(train, epochs = 20, verbose = 1, validation_data = test)

#stop timer
stop_time = datetime.datetime.now()
#print elapsed time
print ("Time required for training:",stop_time - start_time)


 #%%
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Output Section M2
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#assign model values to dictionary
history_dict = estimator.history
#assign training loss values
loss_values = history_dict["loss"]
#assign test loss values
val_loss_values = history_dict["val_loss"]
#calculate epochs
epochs = range(1, len(loss_values) + 1)
#plot training loss
plt.plot(epochs, loss_values, "bo", label="Training loss")
#plot test loss
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
#title plot
plt.title("Training and validation loss M2")
#label x axis
plt.xlabel("Epochs")
#label y axis
plt.ylabel("Loss")
#show legend
plt.legend()
#save figure
plt.savefig('M2_loss.png')
#show plot
plt.show()

#clear plot
plt.clf()
#assign training accuracy values
acc = history_dict["accuracy"]
#assign test accuracy values
val_acc = history_dict["val_accuracy"]
#plot training accuracy values
plt.plot(epochs, acc, "bo", label="Training acc")
#plot test accuracy values
plt.plot(epochs, val_acc, "b", label="Validation acc")
#title plot
plt.title("Training and validation accuracy M2")
#label x axis
plt.xlabel("Epochs")
#label y axis
plt.ylabel("Accuracy")
#show legend
plt.legend()
#save figure
plt.savefig('M2_acc.png')
#show plot
plt.show()

#plot heatmap
labels = ["Mild","Mod","Non","V_Mild"]
y_pred = model2.predict(test)
cf_matrix = tf.math.confusion_matrix(test.labels, tf.argmax(y_pred,1))
sns.heatmap(cf_matrix, annot=True, cmap="Blues",xticklabels=labels, yticklabels=labels)
plt.title("M2 HeatMap")
plt.savefig('M2_heat.png')
plt.show()

#%%
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
*******************************Model 3***********************************
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model Section M3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#augment training data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#rescale test data
test_datagen = ImageDataGenerator(rescale=1./255)

#generate training data
train_gen = train_datagen.flow_from_directory("Alzheimers_Dataset/train",
                                                    class_mode='categorical',
                                                    target_size=(150,150))

#generate test data
test_gen = test_datagen.flow_from_directory("Alzheimers_Dataset/test",
                                                    class_mode='categorical',
                                                    target_size=(150,150))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model Section M3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#start timer
start_time = datetime.datetime.now()

#create feed forward model
model3 = models.Sequential()

random.seed(1)
#Convolution layer with filter size 32, kernel size 3, and input size of 28
model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
#Pooling layer with pool_size 2
model3.add(layers.MaxPooling2D((2, 2)))
#Convolution layer with filter size 64, kernel size 3, and padding so input matches output
model3.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "same"))
#Pooling layer with pool size 2
model3.add(layers.MaxPooling2D((2, 2)))
#Convolution layer with filter size 128, kernel size 3, and padding so input matches output
model3.add(layers.Conv2D(128, (3, 3), activation='relu', padding = "same"))
#Pooling layer with pool size 2
model3.add(layers.MaxPooling2D((2, 2)))
#Flattening layer to convert to 1D
model3.add(layers.Flatten())
#Dense layer to connect to neural network
model3.add(layers.Dense(128, activation='relu'))
#Output layer to classify
model3.add(layers.Dense(4, activation='softmax'))
#Compile model
model3.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#Print summary of model
model3.summary()

#Fit model
estimator = model3.fit(train_gen, epochs = 20, verbose = 1, validation_data = test_gen)

#stop timer
stop_time = datetime.datetime.now()
#print elapsed time
print ("Time required for training:",stop_time - start_time)
 #%%
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Output Section M3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#assign model values to dictionary
history_dict = estimator.history
#assign training loss values
loss_values = history_dict["loss"]
#assign test loss values
val_loss_values = history_dict["val_loss"]
#calculate epochs
epochs = range(1, len(loss_values) + 1)
#plot training loss
plt.plot(epochs, loss_values, "bo", label="Training loss")
#plot test loss
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
#title plot
plt.title("Training and validation loss M3")
#label x axis
plt.xlabel("Epochs")
#label y axis
plt.ylabel("Loss")
#show legend
plt.legend()
#save figure
plt.savefig('M3_loss.png')
#show plot
plt.show()

#clear plot
plt.clf()
#assign training accuracy values
acc = history_dict["accuracy"]
#assign test accuracy values
val_acc = history_dict["val_accuracy"]
#plot training accuracy values
plt.plot(epochs, acc, "bo", label="Training acc")
#plot test accuracy values
plt.plot(epochs, val_acc, "b", label="Validation acc")
#title plot
plt.title("Training and validation accuracy M3")
#label x axis
plt.xlabel("Epochs")
#label y axis
plt.ylabel("Accuracy")
#show legend
plt.legend()
#save figure
plt.savefig('M3_acc.png')
#show plot
plt.show()

#plot heatmap
labels = ["Mild","Mod","Non","V_Mild"]
y_pred = model3.predict(test_gen)
cf_matrix = tf.math.confusion_matrix(test_gen.labels, tf.argmax(y_pred,1))
sns.heatmap(cf_matrix, annot=True, cmap="Blues",xticklabels=labels, yticklabels=labels)
plt.title("M3 HeatMap")
plt.savefig('M3_heat.png')
plt.show()
    