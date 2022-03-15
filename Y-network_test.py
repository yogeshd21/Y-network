import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
#from keras.datasets import cifar100 #Replace use
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop

# load CIFAR dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = cifar100.load_data() #Replace use

num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 3])
x_test = np.reshape(x_test,[-1, image_size, image_size, 3])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


def modl(noconvlyrs, dr, optm):
    # left branch of Y network
    left_inputs = Input(shape=(image_size, image_size, 3))
    x = left_inputs
    filters = 32
    for i in range(noconvlyrs):
        x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', dilation_rate=1)(x)
        x = Dropout(dr)(x)
        x = MaxPooling2D((2,2))(x)
        filters *= 2

    # right branch of Y network
    right_inputs = Input(shape=(image_size, image_size, 3))
    y = right_inputs
    filters = 32
    for i in range(noconvlyrs):
        y = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', dilation_rate=2)(y)
        y = Dropout(dr)(y)
        y = MaxPooling2D((2,2))(y)
        filters *= 2

    y = concatenate([x, y])
    y = Flatten()(y)
    y = Dropout(dr)(y)
    outputs = Dense(num_labels, activation='softmax')(y)

    model = Model([left_inputs, right_inputs], outputs)

    if optm == 'SGD':
        opt = SGD(lr=0.01, momentum=0.9)
    elif optm == 'ADAM':
        opt = Adam(learning_rate=0.01)
    elif optm == 'RMSProp':
        opt = RMSprop(lr=0.01, momentum=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = datagen.flow([x_train, x_train], y_train, batch_size=64)
steps = int(x_train.shape[0] / 64)

df = pd.read_csv('./Data.csv')
for i in range(len(df)):
    model = modl(df['No. of Conv Layers'][i], df['Dropout Rate'][i], df['Optimizer'][i])
    history = model.fit(it_train, steps_per_epoch=steps, epochs=10, validation_data=([x_test, x_test], y_test), verbose=0)
    score = model.evaluate([x_test, x_test], y_test, batch_size=64, verbose=0)
    df.loc[i, 'Accuracy'] = round(score[1] * 100.0, 3)
    print(i, "\nTest accuracy: %.1f%%" % (100.0 * score[1]))

df.to_csv('Ans_CIFAR10.csv', index=False)
#df.to_csv('Ans_CIFAR100.csv', index=False) #Replace use for CIFAR-100 Dataset
