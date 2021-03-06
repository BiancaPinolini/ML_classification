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
        self.kstest_sig = []
        self.kstest_bkg = []
        self.significance_test = []
        self.significance_train = []
        self.precision_test = []
        self.precision_train = []
        self.recall_test = []
        self.recall_train = []
        self.pred_train_temp=[]
        self.pred_test_temp=[]
        self.f1_test = []
        self.f1_train = []
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
        self.acc.append(logs.get('acc')) #training_loss[1])
        self.val_acc.append(logs.get('val_acc'))
        # in newer keras these may be 'accuracy' and 'val_accuracy'
        self.i += 1

        self.pred_test_temp = self.model.predict(self.X_test, batch_size=4096)
        self.pred_train_temp = self.model.predict(self.X_train, batch_size=4096)
        self.pred_test_temp = np.array(self.pred_test_temp).flatten()
        self.pred_train_temp = np.array(self.pred_train_temp).flatten()
        
        # auc_w_test = roc_auc_score(self.y_test, self.pred_test_temp, sample_weight=self.W_test)
        # self.auc_test.append(auc_w_test)
        
        # auc_w_train = roc_auc_score(self.y_train, self.pred_train_temp, sample_weight=self.W_train)
        # self.auc_train.append(auc_w_train)

        kstest_pval_sig = stats.ks_2samp(self.pred_train_temp[self.y_train==1], self.pred_test_temp[self.y_test==1], mode="asymp") # (statistics, pvalue)
        self.kstest_sig.append(kstest_pval_sig[1])
        kstest_pval_bkg = stats.ks_2samp(self.pred_train_temp[self.y_train==0], self.pred_test_temp[self.y_test==0], mode="asymp") # (statistics, pvalue)
        self.kstest_bkg.append(kstest_pval_bkg[1])
        # print("KS test (dnn output: sig (train) vs sig (val))", kstest_pval_sig, ". good: ", kstest_pval_sig[1] > 0.05)
        # print("KS test (dnn output: bkg (train) vs bkg (val))", kstest_pval_bkg, ". good: ", kstest_pval_bkg[1] > 0.05)

        #print(self.y_train.shape, self.y_train[self.y_train==1].shape, self.y_train[self.y_train==0].shape,)
        #print(self.X_train.shape, )
        # s_great_train_mask = (self.y_train==1) & (pred_train[self.y_train==1] > 0.8)
        # print("train", self.X_train.shape, self.y_train.shape, self.W_train.shape, self.pred_train_temp.shape )
        #print("pred", pred_train[self.y_train==1].shape, len(pred_train[self.y_train==1]) )
        #print("W", self.W_train[self.y_train==1].shape)
        #s_tot = np.zeros(len(pred_train))

        TP_train = self.Wnn_train[(self.y_train==1) & (self.pred_train_temp > self.dnncut)].sum()
        FP_train = self.Wnn_train[(self.y_train==0) & (self.pred_train_temp > self.dnncut)].sum() 
        T_train = self.Wnn_train[(self.y_train==1)].sum()
        significance_train =  TP_train  / (np.sqrt( FP_train ))
        self.significance_train.append(significance_train)

        TP_test = self.Wnn_test[(self.y_test==1) & (self.pred_test_temp > self.dnncut)].sum()
        FP_test = self.Wnn_test[(self.y_test==0) & (self.pred_test_temp > self.dnncut)].sum()
        T_test = self.Wnn_test[(self.y_test==1)].sum()
        significance_test =  TP_test  / (np.sqrt( FP_test ))
        self.significance_test.append(significance_test)

        self.precision_test.append(TP_test/ (TP_test+FP_test))
        self.precision_train.append(TP_train/ (TP_train+FP_train))
        self.recall_test.append(TP_test/ T_test)
        self.recall_train.append(TP_train/ T_train)
        
        self.f1_train.append( 2* (self.precision_train[-1] * self.recall_train[-1])/ (self.precision_train[-1] + self.recall_train[-1]) )
        self.f1_test.append( 2* (self.precision_test[-1] * self.recall_test[-1])/ (self.precision_test[-1] + self.recall_test[-1]) )

    def performance_plot(self):
        self.figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24,10))

        # ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, "o-", label="loss (train)")
        ax1.plot(self.x, self.val_losses, "o-", label="loss (val)")
        ax1.set_yscale("log")
        ax1.set_xlabel("epochs")
        ax1.legend()

        ax2.plot(self.x, self.f1_train, "o-", label="F1 score (train) thr0.85")
        ax2.plot(self.x, self.f1_test, "o-", label="F1 score (test) thr0.85")
        ax2.set_xlabel("epochs")
        ax2.legend()
        
        ax3.plot(self.x, self.precision_train, "o-", label="Precision (train) thr0.85")
        ax3.plot(self.x, self.precision_test, "o-", label="Precision (test) thr0.85")
        ax3.set_xlabel("epochs")
        ax3.legend()

        ax5.plot()       

        bins=25
        ax4.hist(self.pred_train_temp[self.y_train==0],weights=self.W_train[self.y_train==0], bins=bins, range=(0.,1.), density=True, label="bkg (train)", histtype="step")
        ax4.hist(self.pred_train_temp[self.y_train==1],weights=self.W_train[self.y_train==1], bins=bins, range=(0.,1.), density=True, label="sig (train)", histtype="step")
        dnnout_false = ax4.hist(self.pred_test_temp[self.y_test==0],weights=self.W_test[self.y_test==0], bins=bins, range=(0.,1.), density=True, label="bkg (val)", histtype="step")
        dnnout_true  = ax4.hist(self.pred_test_temp[self.y_test==1],weights=self.W_test[self.y_test==1], bins=bins, range=(0.,1.), density=True, label="sig (val)", histtype="step")
        ax4.set_xlabel("DNN output")
        ax4.legend()


        ax6.plot(self.x, self.recall_train, "o-", label="Recall (train) thr0.85")
        ax6.plot(self.x, self.recall_test, "o-", label="Recall (test) thr0.85")
        ax6.set_xlabel("epochs")
        ax6.legend()
        # ax6.plot(self.x, self.significance_train, "o-", color="blue")
        # ax6.set_ylabel("S / sqrt(B) (train)", color='blue')
        # ax6.set_xlabel("epochs")
        # ax7 = ax6.twinx()
        # ax7.plot(self.x, self.significance_test, "o-", color="orange")
        # ax7.set_ylabel("S / sqrt(B) (val)", color='orange')

        if not self.batch_mode:
            clear_output(wait=True)
            plt.show()

    def save_figure(self, fname):
        if self.batch_mode:
            self.performance_plot()
        self.figure.savefig(fname)
        plt.close(self.figure)
