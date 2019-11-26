# -*- coding: utf-8 -*-‘
"""

"""

from time import time
import numpy as np
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import metrics
from DenPeakcode import DenPeakCluster
K.set_image_data_format('channels_last')


def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class DEC(object):
    def __init__(self,
                 dims,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.pretrained = False
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256,
                 save_dir='results/temp', verbose=1, aug_ae=False):
        print('Pretraining......')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]

        # begin pretraining
        t0 = time()
        if not aug_ae:
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=verbose)
        else:
            print('-=*'*20)
            print('Using augmentation for ae')
            print('-=*'*20)
            def gen(x, batch_size):
                if len(x.shape) > 2:  # image
                    gen0 = self.datagen.flow(x, shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        yield [batch_x, batch_x]
                else:
                    width = int(np.sqrt(x.shape[-1]))
                    if width * width == x.shape[-1]:  # gray
                        im_shape = [-1, width, width, 1]
                    else:  # RGB
                        width = int(np.sqrt(x.shape[-1] / 3.0))
                        im_shape = [-1, width, width, 3]
                    gen0 = self.datagen.flow(np.reshape(x, im_shape), shuffle=True, batch_size=batch_size)
                    while True:
                        batch_x = gen0.next()
                        batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
                        yield [batch_x, batch_x]
            self.autoencoder.fit_generator(gen(x, batch_size), steps_per_epoch=int(x.shape[0]/batch_size),
                                           epochs=epochs, callbacks=cb, verbose=verbose,
                                           workers=8, use_multiprocessing=True)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def extract_features(self, x):
        return self.encoder.predict(x)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def random_transform(self, x):
        if len(x.shape) > 2:  # image
            return self.datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()

        # if input a flattened vector, reshape to image before transform
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen = self.datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=x.shape[0])
        return np.reshape(gen.next(), x.shape)

    def add_noise(self, x):
        noise = 0.05 * np.random.randn(*x.shape)
        mask = x != 0
        return x + noise * mask

    def fit(self, x, y=None,save_dir='./results/temp'):
        # print('Begin training:', '-' * 60)

        t1 = time()
        print('******************** Use Denpeak to Cluster ************************')


        features = self.encoder.predict(x)
        print("features shape:", features.shape)
        features = TSNE(n_components=2).fit_transform(features)
        # np.savetxt("features.txt", features)
        print("features shape:",features.shape)
        y_pred, y_border, center_num ,dc_percent, dc= DenPeakCluster(features)
        print('saving picture to:', save_dir+'/2D.png')
        plt.cla()
        plt.scatter(features[:, 0], features[:, 1], c=y_pred, s=0.5, alpha=0.5)
        plt.savefig(save_dir+'/2D.png')
        np.savetxt(save_dir+'/dc_coeff.txt', [dc_percent, dc])

        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss', 'center_num'])
        logwriter.writeheader()

        acc = np.round(metrics.acc(y, y_pred), 5)
        nmi = np.round(metrics.nmi(y, y_pred), 5)
        ari = np.round(metrics.ari(y, y_pred), 5)
        # if acc>=0.95:
        np.savetxt(save_dir+'/features.txt', features)
        np.savetxt(save_dir+'/labels.txt', y_pred)
        np.savetxt(save_dir+'/border.txt', y_border)
        from Draw_border import draw
        draw(save_dir)
        logdict = dict(iter=0, acc=acc, nmi=nmi, ari=ari, center_num=center_num)
        logwriter.writerow(logdict)
        logfile.flush()
        print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f; center_num=%d' % (0, acc, nmi, ari, center_num))
        logfile.close()

        return y_pred

