from ConvDEC import ConvDEC
import os, csv
from datasets import load_data_conv
from keras.initializers import VarianceScaling
import numpy as np
from time import time


def run_exp(dbs, aug_ae,expdir, ae_weights_dir, trials=5, verbose=0):
    # Log files
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logfile = open(expdir + '/results.csv', 'a')
    logwriter = csv.DictWriter(logfile, fieldnames=['trials', 'acc', 'nmi', 'ari', 'center_num', 'time'])
    logwriter.writeheader()

    # Begin training on different datasets
    for db in dbs:
        logwriter.writerow(dict(trials=db, acc='', nmi='', ari='', center_num='', time=''))
        save_db_dir = os.path.join(expdir, db)
        if not os.path.exists(save_db_dir):
            os.makedirs(save_db_dir)

        # load dataset
        x, y = load_data_conv(db)

        # setting parameters
        # n_clusters = len(np.unique(y))
        # update_interval = 140 if db in ['fmnist', 'mnist'] else 30
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')  # sqrt(1./fan_in)

        # Training
        results = np.zeros(shape=[2, trials, 5], dtype=float)  # init metrics before finetuning
        for i in range(trials):  # base
            t0 = time()
            save_dir = os.path.join(save_db_dir, 'trial%d' % i)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],
                            init=init)
            # model.compile(optimizer='adam', loss='kld')

            # whether to use pretrained ae weights
            if ae_weights_dir is None:
                model.pretrain(x, y, optimizer='adam', epochs=500,
                               save_dir=save_dir, verbose=verbose, aug_ae=aug_ae)
            else:
                model.autoencoder.load_weights(os.path.join(ae_weights_dir, db, 'trial%d' % i, 'ae_weights.h5'))
            t1 = time()

            # training
            model.fit(x, y=y, save_dir=save_dir)

            # saving log results
            log = open(os.path.join(save_dir, 'log.csv'), 'r')
            reader = csv.DictReader(log)
            metrics = []
            for row in reader:
                metrics.append([row['acc'], row['nmi'], row['ari'], row['center_num']])
            results[0, i, :] = np.asarray(metrics[0] + [t1-t0])
            results[1, i, :] = np.asarray(metrics[-1] + [time()-t0])
            log.close()

        # save all results to `results.csv`
        # results.shape = [2, trials, 5], where results[0] is initial and results[1] is final clustering performance
        for result in results:
            for t, line in enumerate(result):
                logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], ari=line[2], center_num=line[3],time=line[4]))
            mean = np.mean(result, 0)
            logwriter.writerow(dict(trials=' ', acc=mean[0], nmi=mean[1], ari=mean[2], center_num='', time=mean[4]))
            logfile.flush()

    logfile.close()


if __name__=="__main__":
    # Global experiment settings
    trials = 5
    verbose = 0
    dbs = ['LetterA_J','usps10k', 'mnist-test', 'fashion_test']  # 'stl', 'mnist', 'fmnist'
    """exp1: ConvDEC-augae0"""
    run_exp(dbs, aug_ae=False,
            expdir='results/exp1-convdec-augae0',
            ae_weights_dir=None,
            verbose=verbose, trials=trials)
    """exp2: ConvDEC-augae1"""
    run_exp(dbs, aug_ae=True,
            expdir='results/exp2-convdec-augae1',
            ae_weights_dir=None,
            verbose=verbose, trials=trials)
