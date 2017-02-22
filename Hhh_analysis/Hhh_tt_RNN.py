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
#import deepdish.io as io #not available yet
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
#io.save(open('ttbar.h5', 'wb'), ttbar)
#new_df = io.load(open('ttbar.h5', 'rb'))
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

# Recurrent Neural Networks
# Let's look at a fancier way of solving the same classification problem. In this case we will use Recurrent Neural Netwroks. These allow you to process variable length sequences of data.
# For example, we can use them to describe an event in terms of the properties of its jets, whose number varies event by event.
# We could also describe the same event using the properties of its muons, or any other particle that appears in it.
# Because the number of particles of each type changes in each event, we need the flexibility of RNNs to process this type of data.
DRvars = [key for key in df.keys() if( key.startswith('dR_') or key.startswith('MINdR') or key.startswith('dphi') )]
print DRvars
MASSvars = [key for key in df.keys() if( key.startswith('mass') or key.startswith('MT2') )]
print MASSvars
PTvars = [key for key in df.keys() if key.startswith('pt_b1b2')]
print PTvars
df_dr = df[DRvars].copy()
df_mass = df[MASSvars].copy()
df_pt = df[PTvars].copy()
#num_dr   = max([len(e) for e in df_dr.Electron_E])
#num_mass = max([len(m) for m in df_mass.Muon_E])
#num_pt   = max([len(gam) for gam in df_pt.Photon_E])

# Since I'm not going to retrain this net right now but I'm just loading in some pre-trained weights, I will also need to load the exact ordering of indices that I used the time I trained.
# This way, I will ensure that the random shuffling in train_test_split will not cause me to evaluate on a subset of examples I previously trained on.
##ix_train, ix_test = cPickle.load(open('./models/ixtraintest.pkl', 'rb'))
##X_train, y_train, w_train = X[ix_train], y[ix_train], w[ix_train]
##X_test, y_test, w_test = X[ix_test], y[ix_test], w[ix_test]
## Just for the sake of variety, you can either have class labels like (0, 1, 2, 3, ...) and train using spare_categorical_crossentropy as you loss function -- like we did before -- or, equivalently,
## you can have class labels like ([1, 0, 0, 0, ...], [0, 1, 0, 0, ...], ...) and train using categorical_crossentropy.
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
#
#Xjet_train, Xjet_test = create_stream(df_jets, num_jets, sort_col='Jet_btag')
#Xphoton_train, Xphoton_test = create_stream(df_photons, num_photons, sort_col='Photon_E')
#Xmuon_train, Xmuon_test = create_stream(df_muons, num_muons, sort_col='Muon_E')
#Xelectron_train, Xelectron_test = create_stream(df_electrons, num_electrons, sort_col='Electron_E')
#jet_channel = Sequential()
#muon_channel = Sequential()
#electron_channel = Sequential()
#photon_channel = Sequential()
#JET_SHAPE = Xjet_train.shape[1:]
#MUON_SHAPE = Xmuon_train.shape[1:]
#ELECTRON_SHAPE = Xelectron_train.shape[1:]
#PHOTON_SHAPE = Xphoton_train.shape[1:]
#jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
#jet_channel.add(GRU(25, name='jet_gru'))
#jet_channel.add(Dropout(0.3, name='jet_dropout'))
#
#muon_channel.add(Masking(mask_value=-999, input_shape=MUON_SHAPE, name='muon_masking'))
#muon_channel.add(GRU(10, name='muon_gru'))
#muon_channel.add(Dropout(0.3, name='muon_dropout'))
#
#electron_channel.add(Masking(mask_value=-999, input_shape=ELECTRON_SHAPE, name='electron_masking'))
#electron_channel.add(GRU(10, name='electron_gru'))
#electron_channel.add(Dropout(0.3, name='electron_dropout'))
#
#photon_channel.add(Masking(mask_value=-999, input_shape=PHOTON_SHAPE, name='photon_masking'))
#photon_channel.add(GRU(10, name='photon_gru'))
#photon_channel.add(Dropout(0.3, name='photon_dropout'))
#combined_rnn = Sequential()
#
#combined_rnn.add(Merge([
#            jet_channel, muon_channel, electron_channel, photon_channel
#        ], mode='concat'))
#
#combined_rnn.add(Dense(64, activation = 'relu'))
#combined_rnn.add(Dropout(0.3))
#combined_rnn.add(Highway(activation = 'relu'))
#combined_rnn.add(Dropout(0.3))
#combined_rnn.add(Highway(activation = 'relu'))
#combined_rnn.add(Dropout(0.3))
#combined_rnn.add(Dense(3, activation='softmax'))
#combined_rnn.summary()
#combined_rnn.compile('adam', 'categorical_crossentropy')
#print 'Training RNN:'
#try:
#    combined_rnn.fit([Xjet_train, Xmuon_train, Xelectron_train, Xphoton_train], y_train, batch_size=16,
#            class_weight={
#                0 : 0.33 * (float(len(y)) / (y == 0).sum()),
#                1 : 0.33 * (float(len(y)) / (y == 1).sum()),
#                2 : 0.33 * (float(len(y)) / (y == 2).sum())
#        },
#        callbacks = [
#            EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
#            ModelCheckpoint('./models/combinedrnn_tutorial-progress', monitor='val_loss', verbose=True, save_best_only=True)
#        ],
#    nb_epoch=30, 
#    validation_split = 0.2) 
#
#except KeyboardInterrupt:
#    print 'RNN training ended early.'
## -- load in best network
#combined_rnn.load_weights('./models/combinedrnn_tutorial-progress')
#print 'Saving RNN weights...'
#combined_rnn.save_weights('./models/combinedrnn_tutorial.h5', overwrite=True)
#
#json_string = combined_rnn.to_json()
#open('./models/combinedrnn_tutorial.json', 'w').write(json_string)
#yhat_rnn = combined_rnn.predict([Xjet_test, Xmuon_test, Xelectron_test, Xphoton_test], verbose = True, batch_size = 512)
#bins = np.linspace(0,3,4)
#fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
#_ = plt.hist(np.argmax(yhat_rnn, axis=1), bins=bins, histtype='stepfilled', alpha=0.5, label='prediction', weights=w_test)
#_ = plt.hist(y_test.argmax(axis=1), bins=bins, histtype='stepfilled', alpha=0.5, label='truth', weights=w_test)
#plt.legend(loc='upper left')
##plt.show()
#print('Saving:',folder + '/PerformanceRNN.png')
#plt.savefig(folder + '/PerformanceRNN.png')
## -- turn the predictions back into class labels
#yhat_rnn_cls = np.argmax(yhat_rnn, axis=1)
#yhat_rnn_cls
## -- do the same for the truth labels
#y_test_cls = np.argmax(y_test, axis=1)
#print 'Signal efficiency:', w_test[(y_test_cls == 0) & (yhat_rnn_cls == 0)].sum() / w_test[y_test_cls == 0].sum()
#b_eff = w_test[(y_test_cls != 0) & (yhat_rnn_cls == 0)].sum() / w_test[y_test_cls != 0].sum()
#print 'Background efficiency:', b_eff
#print 'Background rej:', 1 / b_eff
#
