import os
import numpy as np
from pylab import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def draw(path):
    features=np.loadtxt(path+'/features.txt')
    y_pred=np.loadtxt(path + '/labels.txt')
    y_border=np.loadtxt(path + '/border.txt')
    plt.cla()
    plt.scatter(features[:, 0], features[:, 1], c=y_pred, s=0.5, alpha=0.5)
    idx_1 = np.where(y_border == -1)
    plt.scatter(features[idx_1, 0], features[idx_1, 1], c='k', marker='x', s=0.5, alpha=0.5)
    print('saving picture to:', path + '/2D_border.png')
    plt.savefig(path+'/2D_border.png')