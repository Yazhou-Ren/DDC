from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, InputLayer
from keras.models import Sequential, Model
from keras.initializers import VarianceScaling
from keras.preprocessing.image import ImageDataGenerator

from DEC import DEC


def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    model.add(InputLayer(input_shape))
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1'))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    encoder = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
    return model, encoder


class ConvDEC(DEC):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 init='glorot_uniform'):

        # self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
        self.datagenx = ImageDataGenerator()
        self.autoencoder, self.encoder = CAE(input_shape, filters)


