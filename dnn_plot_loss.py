'''
keras callback to plot loss
'''

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import keras
from sklearn.metrics import roc_auc_score, roc_curve

from scipy import stats

## callbacks
# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, model, data, dnncut=0.85, batch_mode=False):
        self.model = model
        self.X_train = data["X_train"]
        self.X_test = data["X_val"]
        self.y_train = data["y_train"]
        self.y_test = data["y_val"]
        self.W_train = data["W_train"]
        self.W_test = data["W_val"]  # use the validation data for plots
        self.Wnn_train = data["Wnn_train"]
        self.Wnn_test = data["Wnn_val"]
        self.batch_mode = batch_mode
        self.dnncut = dnncut

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.dnn_score_plot = []
        self.dnn_score_log = []
        self.significance_test = []
        self.significance_train = []
        self.precision_test = []
        self.precision_train = []
        self.recall_test = []
        self.recall_train = []
        self.pred_train_temp=[]
        self.pred_test_temp=[]
        self.figure = None
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.performance_save(logs)
        if not self.batch_mode:
            self.performance_plot()

    def performance_save(self, logs):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss')) #training_loss[0])
        self.val_losses.append(logs.get('val_loss'))
        # 'acc' and 'val_acc' work in "96 python3" in swan.cern.ch
        self.acc.append(logs.get('accuracy')) #training_loss[1])
        self.val_acc.append(logs.get('val_accuracy'))
        # in newer keras these may be 'accuracy' and 'val_accuracy'
        self.i += 1

        self.pred_test_temp = self.model.predict(self.X_test, batch_size=4096)
        self.pred_train_temp = self.model.predict(self.X_train, batch_size=4096)
        self.pred_test_temp = np.array(self.pred_test_temp).flatten()
        self.pred_train_temp = np.array(self.pred_train_temp).flatten()

        self.precision_test.append(TP_test/ (TP_test+FP_test))
        self.precision_train.append(TP_train/ (TP_train+FP_train))
        self.recall_test.append(TP_test/ T_test)
        self.recall_train.append(TP_train/ T_train)

    def performance_plot(self):
        self.figure, ((ax1, ax2, ax3), (ax4, , ax6)) = plt.subplots(2, 3, figsize=(24,10))

        # ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, "o-", label="loss (train)")
        ax1.plot(self.x, self.val_losses, "o-", label="loss (val)")
        ax1.set_yscale("log")
        ax1.set_xlabel("epochs")
        ax1.legend()

        ax2.plot(self.x, self.acc, "o-", label="accuracy (train)")
        ax2.plot(self.x, self.val_acc, "o-", label="accuracy (val)")
        ax2.set_xlabel("epochs")
        ax2.legend()
        
        ax3.plot(self.x, self.precision_train, "o-", label="Precision (train) thr0.85")
        ax3.plot(self.x, self.precision_test, "o-", label="Precision (test) thr0.85")
        ax3.set_xlabel("epochs")
        ax3.legend()

        bins=25
        ax4.hist(self.pred_train_temp[self.y_train==0],weights=self.W_train[self.y_train==0], bins=bins, range=(0.,1.), density=True, label="bkg (train)", histtype="step")
        ax4.hist(self.pred_train_temp[self.y_train==1],weights=self.W_train[self.y_train==1], bins=bins, range=(0.,1.), density=True, label="sig (train)", histtype="step")
        dnnout_false = ax4.hist(self.pred_test_temp[self.y_test==0],weights=self.W_test[self.y_test==0], bins=bins, range=(0.,1.), density=True, label="bkg (val)", histtype="step")
        dnnout_true  = ax4.hist(self.pred_test_temp[self.y_test==1],weights=self.W_test[self.y_test==1], bins=bins, range=(0.,1.), density=True, label="sig (val)", histtype="step")
        ax4.set_xlabel("DNN score")
        ax4.legend()

        ax6.plot(self.x, self.recall_train, "o-", label="Recall (train) thr0.85")
        ax6.plot(self.x, self.recall_test, "o-", label="Recall (test) thr0.85")
        ax6.set_xlabel("epochs")
        ax6.legend()

        if not self.batch_mode:
            clear_output(wait=True)
            plt.show()

    def save_figure(self, fname):
        if self.batch_mode:
            self.performance_plot()
        self.figure.savefig(fname)
        plt.close(self.figure)
