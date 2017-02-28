# https://github.com/YaleATLAS/CERNDeepLearningTutorial/blob/master/FFNN_RNN.ipynb
import os, ROOT, sys, array, re, math, random, subprocess
from ROOT import gSystem, gROOT, gApplication, TFile, TTree, TCut, TH1F
from math import *
import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array, root2array
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Highway, MaxoutDense
from keras.layers import Masking, GRU, Merge
from keras.layers import Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
import cPickle
import deepdish.io as io
execfile("Useful_func.py")
# Fix random seed for reproducibility
seed = 7; np.random.seed(seed);

debug=True
folder='Plots/'
MakePlots=True
folderCreation  = subprocess.Popen(['mkdir -p ' + folder], stdout=subprocess.PIPE, shell=True); folderCreation.communicate()
folderCreation2 = subprocess.Popen(['mkdir -p models/'], stdout=subprocess.PIPE, shell=True); folderCreation2.communicate()

# Pre-selection and branches to include
my_selec = 'MMC_h2massweight1_prob>200 && hasRECOjet1 && hasRECOjet1 && hasMET && hastwomuons && (((b1jet_btag&2)>0 && (b2jet_btag&3)>0) || ((b1jet_btag&3)>0 && (b 2jet_btag&2)>0)) && dR_l1l2<3.3 && dR_l1l2>0.07 && dR_b1b2<5. && mass_l1l2<100 && mass_l1l2>5. && mass_b1b2>150 && dR_bl<5 && dR_l1l2b1b2<6 && MINdR_bl<3.2 && MINdR_bl>0.4 && mass_b1b2<700 && mass_trans<250 && MT2<400 && pt_b1b2<300'
my_branches_training = ["dR_l1l2","dR_b1b2","dR_bl","dR_l1l2b1b2","MINdR_bl","dphi_l1l2b1b2","mass_l1l2","mass_b1b2","mass_trans","MT2","pt_b1b2"]
my_branches = my_branches_training; my_branches.append("weight"); my_branches.append("reweighting");
my_branches_training = ["dR_l1l2","dR_b1b2","dR_bl","dR_l1l2b1b2","MINdR_bl","dphi_l1l2b1b2","mass_l1l2","mass_b1b2","mass_trans","MT2","pt_b1b2"] # Not sure why but otherwise it has weights
# File to dataframe
Hhh3 = root2panda('files_root/delphes_B3_1M_PU40ALL_1Ag_mvammc.root', 'evtree', branches=my_branches, selection=my_selec)
ttbar = root2panda('files_root/delphes_tt_4M_PU40_WtomuALL_1Ag_mvammc.root', 'evtree', branches=my_branches, selection=my_selec)
# Create a variable that is the total weight
Hhh3['fin_weight']  = Hhh3['weight']*Hhh3['reweighting']
ttbar['fin_weight'] = ttbar['weight']*ttbar['reweighting']
# Save the dataframe as h5 file (for quick loading in the future).
#io.save(open('models/ttbar.h5', 'wb'), ttbar_0)
#ttbar = io.load(open('models/ttbar.h5', 'rb'))

if debug:
  print("---> Hhh3 Displayed as panda dataframe: "); print(Hhh3)
  print("The shape for Hhh3 is [nb_events, nb_variables]: "); print(Hhh3.shape)
  print("The shape for tt is [nb_events, nb_variables]: "); print(ttbar.shape)
# Plots of the branches we selected
if (MakePlots and False) :
  for key in ttbar.keys():
      if(key!="weight" and key!="reweighting" and key!="fin_weight") :
        matplotlib.rcParams.update({'font.size': 16})
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
        bins = np.linspace(my_max(min(ttbar[key]),0.), max(ttbar[key]), 50)
        _ = plt.hist(Hhh3[key],  bins=bins, histtype='step', normed=True, label=r'$Hhh (B3)$', linewidth=2)
        _ = plt.hist(ttbar[key], bins=bins, histtype='step', normed=True, label=r'$t\overline{t}$')
        plt.xlabel(key)
        plt.ylabel('Entries')
        plt.legend(loc='best')
        print('Saving:',folder + '/' + str(key) + '.png')
        plt.savefig(folder + '/' + str(key) + '.png')

print('Now lets start to talk about DNN!')
df =  pd.concat((Hhh3[my_branches_training], ttbar[my_branches_training]), ignore_index=True)
# Turn the df the desired ndarray "X" that can be directly used for ML applications.
X = df.as_matrix() # Each row is an object to classify, each column corresponds to a specific variable.
# Take the weights
w =  pd.concat((Hhh3['fin_weight'], ttbar['fin_weight']), ignore_index=True).values
# This is the array with the true values
y = []
for _df, ID in [(Hhh3, 0), (ttbar, 1)]:
    y.extend([ID] * _df.shape[0])
y = np.array(y)

# Randomly shuffle and automatically split all your objects into train and test subsets
ix = range(X.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks
X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=0.7) # Train here is 70% of the total statistic
# It is common practice to scale the inputs to Neural Nets such that they have approximately similar ranges.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Multilayer Perceptron (MLP) definition
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu')) # Linear transformation of the input vector. The first number is output_dim.
model.add(Dropout(0.2)) # To avoid overfitting. It masks the outputs of the previous layer such that some of them will randomly become inactive and will not contribute to information propagation.
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Highway(activation='relu')) # Use adaptive gating units which learn to regulate the flow of information through a network. Improves ability to train very deep feed-forward nets.
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(MaxoutDense(7, 4)) # A Dense layer that learns its own activation function.
model.add(Dense(2, activation='softmax')) # Last layer has to have the same dimensionality as the number of classes we want to predict, here 2.
model.summary()
# Now you need to declare what loss function and optimizer to use (and compile your model).
model.compile('adam', 'sparse_categorical_crossentropy')
print('---------------------------Training:---------------------------')
try:
    model.fit(X_train, y_train, class_weight={
                0 : 0.5 * (float(len(y)) / (y == 0).sum()), # (y == 0).sum() give the total number of element that are zero in y (array of the true category).
                1 : 0.5 * (float(len(y)) / (y == 1).sum())
        },
        callbacks = [
            EarlyStopping(verbose=True, patience=6, monitor='val_loss'),
            ModelCheckpoint('./models/tutorial-progress.h5', monitor='val_loss', verbose=True, save_best_only=True)
        ],
    nb_epoch=10, 
    validation_split = 0.2) 

except KeyboardInterrupt:
    print 'Training ended early.'

# Load in best network
model.load_weights('./models/tutorial-progress.h5')
print 'Saving weights...'
model.save_weights('./models/tutorial.h5', overwrite=True)
json_string = model.to_json()
open('./models/tutorial.json', 'w').write(json_string)
print 'Testing...'
yhat = model.predict(X_test, verbose = True, batch_size = 512) # yhat is the categorization given by the DNN
#Turn them into classes
yhat_cls = np.argmax(yhat, axis=1) # Returns the indices of the maximum values along axis 1
bins = np.linspace(-0.5,1.5,3)
names = ['','Hhh B3','','tt']
fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
ax = plt.subplot()
ax.set_xticklabels(names, rotation=45)
_ = plt.hist(yhat_cls, bins=bins, histtype='stepfilled', alpha=0.5, label='prediction',log=True)#, weights=w_test)#,log=True)
_ = plt.hist(y_test, bins=bins, histtype='stepfilled', alpha=0.5, label='truth',log=True)#, weights=w_test)#,log=True)
plt.legend(loc='upper right')
print('Saving:',folder + '/Performance.png')
plt.savefig(folder + '/Performance.png')

print 'Signal efficiency:',     w_test[(y_test == 0) & (yhat_cls == 0)].sum() / w_test[y_test == 0].sum()
print 'Background efficiency:', w_test[(y_test != 0) & (yhat_cls == 0)].sum() / w_test[y_test != 0].sum()

