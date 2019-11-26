import numpy as np
import os

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_mnist_test():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    _, (x, y) = mnist.load_data()
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('mnist_test samples', x.shape)
    return x, y

def load_fashion_mnist():
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('Fashion MNIST samples', x.shape)
    return x, y

def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16, 16, 1])
    print('USPS samples', x.shape)
    return x, y

def load_cifar10():
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))
    x = x / 255.0
    print('cifar10 samples', x.shape)
    return x, y

def load_stl(data_path='./data/stl', target_size=96):
    assert os.path.exists(data_path + '/stl_features.npy') or os.path.exists(data_path + '/train_X.bin'), \
        "No data! Use %s/get_data.sh to get data ready, then come back" % data_path

    # get labels
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))

    # get data
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
    x1 = x1.reshape((int(x1.size / 3 / 96 / 96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    x2 = x2.reshape((int(x2.size / 3 / 96 / 96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x = np.concatenate((x1, x2)).astype(float)

    if target_size != 96:
        from keras.preprocessing.image import img_to_array, array_to_img
        x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((target_size, target_size))) for im in x])
    print('stl samples', x.shape)
    return x / 255.0, y



def load_h5_dataset(dataset_path):
    import h5py
    hf = h5py.File(dataset_path + '/data4torch.h5', 'r')
    X = np.asarray(hf.get('data'), dtype='float32')
    X_train = X / np.float32(255.0)
    X_train = np.transpose(X_train, [0, 2, 3, 1])
    y_train = np.asarray(hf.get('labels'), dtype='int32')
    print(dataset_path ,':', X_train.shape)
    return X_train, y_train

def load_fashion_test():
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = x_test
    y = y_test
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('Fashion TEST samples', x.shape)
    return x, y

def load_LetterA_J():
    x=np.loadtxt('./data/LetterA_J/images.test.txt')
    y=np.loadtxt('./data/LetterA_J/labels.test.txt')
    x=x.reshape([-1,28,28,1])
    print("LetterA_J samples", x.shape)
    x=x/255.0
    return x,y

def load_data_conv(dataset):
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fmnist':
        return load_fashion_mnist()
    elif dataset == 'usps10k':
        return load_usps()
    elif dataset == 'fashion_test':
        return load_fashion_test()
    elif dataset == 'LetterA_J':
        return load_LetterA_J()
    elif dataset == 'stl':
        return load_stl()
    elif dataset == 'cifar10':
        return load_cifar10()
    elif dataset in ['FRGC', 'UMist', 'CMU-PIE', 'COIL-20']:
        return load_h5_dataset('./data//JULE-Torch/datasets/' + dataset)
    else:
        print('Not defined for loading', dataset)
        exit(0)


