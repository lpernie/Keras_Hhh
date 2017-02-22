# https://github.com/YaleATLAS/CERNDeepLearningTutorial/blob/master/FFNN_RNN.ipynb
import ROOT, array, os, sys, re, math, random
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

# Import root file (you can also specify which branches)
debug=True
folder='Plots/'
MakePlots=False
ttbar = root2array('files/ttbar.root')
if debug:
  # Display your newly created object
  print("---> The whole root file is:"); print(ttbar);
  print("---> The file type is: "); print(type(ttbar))
  print("---> The events are: "); print(ttbar.shape)
  print("---> The branches names are: "); print(ttbar.dtype.names)
# Turn an ndarray into a pandas dataframe
df = pd.DataFrame(ttbar)
if debug:
  # Now it is much better displayed
  print("---> Displayed as panda dataframe: "); print(df)
  print("The shape is [nb_events, nb_variables]: "); print(df.shape)
# Now I di the same for this other root file, but I use a function I created
singletop = root2panda('files/single_top.root', 'events')
# You need deepdish.io  that is not installed yet
#io.save(open('ttbar.h5', 'wb'), df)
#new_df = io.load(open('ttbar.h5', 'rb'))

# We can cut the original dataframe in such a way that we only kkep jets informations.
print [key for key in df.keys() if key.startswith('Jet')]
jet_df = df[[key for key in df.keys() if key.startswith('Jet')]]
if debug:
  # Now check that each entry is a jet.
  print("---> Displayed as panda dataframe with only jets information: "); print(jet_df)
# And if we want to train the jets, we want a dataframe that is jet-flat and not event-flat
df_flat = pd.DataFrame({k: flatten(c) for k, c in jet_df.iteritems()})
if debug:
  # Now check that the dataframe is jet-flat
  print("---> Displayed as panda dataframe jet-flat: "); print(df_flat)
# We can inspect the variables super quickly
if MakePlots :
  for key in df_flat.keys():
      matplotlib.rcParams.update({'font.size': 16})
      fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
      bins = np.linspace(min(df_flat[key]), max(df_flat[key]), 30)
      _ = plt.hist(df_flat[key], bins=bins, histtype='step', label=r'$t\overline{t}$')
      plt.xlabel(key)
      plt.ylabel('Number of Jets')
      plt.legend()
      #plt.plot()
      print('Saving:',folder + '/' + str(key) + '.png')
      plt.savefig(folder + '/' + str(key) + '.png')
# And we can create extra variables
df['Jet_P'] = (df['Jet_Px']**2 + df['Jet_Py']**2 + df['Jet_Pz']**2)**(0.5)
# You can easily slice dataframes by specifying the names of the branches you would like to select 
if False:
  print(df[['Jet_Px', 'Jet_Py', 'Jet_Pz', 'Jet_E']])
# You can also build four vectors and store them in a new column in 1 line of code 
from rootpy.vector import LorentzVector
df['Jet_4V'] = [map(lambda args: LorentzVector(*args), zip(px, py, pz, e)) for 
                (_, (px, py, pz, e)) in df[['Jet_Px', 'Jet_Py', 'Jet_Pz', 'Jet_E']].iterrows()]

# Now lets talk about DNN
print('Now lets start to talk about DNN!')
# MC signal:
ttbar = root2panda('files/ttbar.root', 'events') 
# MC backgrounds:
dy = root2panda('files/dy.root', 'events')
wj = root2panda('files/wjets.root', 'events')
ww = root2panda('files/ww.root', 'events')
wz = root2panda('files/wz.root', 'events')
zz = root2panda('files/zz.root', 'events')
singletop = root2panda('files/single_top.root', 'events')
qcd = root2panda('files/qcd.root', 'events')
# data:
data = root2panda('files/data.root', 'events')
# Imagine that all samples have passed a trigger, and in tt you have the trigger info (True or False) and you want to cut on it, selecting only the one that passed
ttbar = ttbar[ttbar['triggerIsoMu24']].reset_index(drop=True)
# We want to start by training a simple model that only relies on event-level variables
npart = ['NJet', 'NMuon', 'NElectron', 'NPhoton', 'MET_px', 'MET_py']
# Make some plots
if MakePlots :
  for key in npart:
      # Font and canvas size
      matplotlib.rcParams.update({'font.size': 16})
      fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
      # Declare common binning strategy
      bins=np.linspace(min(ttbar[key]), max(ttbar[key]) + 1, 30)
      _ = plt.hist(ttbar[key], histtype='step', normed=False, bins=bins, weights=ttbar['EventWeight'], label=r'$t\overline{t}$', linewidth=2)
      _ = plt.hist(dy[key], histtype='step', normed=False, bins=bins, weights=dy['EventWeight'], label='Drell Yan')
      _ = plt.hist(wj[key], histtype='step', normed=False, bins=bins, weights=wj['EventWeight'], label=r'$W$ + jets')
      _ = plt.hist(ww[key], histtype='step', normed=False, bins=bins, weights=ww['EventWeight'], label=r'$WW$')
      _ = plt.hist(wz[key], histtype='step', normed=False, bins=bins, weights=wz['EventWeight'], label=r'$WZ$')
      _ = plt.hist(zz[key], histtype='step', normed=False, bins=bins, weights=zz['EventWeight'], label=r'$ZZ$')
      _ = plt.hist(singletop[key], histtype='step', normed=False, bins=bins, weights=singletop['EventWeight'], label=r'single $t$')
      _ = plt.hist(qcd[key], histtype='step', normed=False, bins=bins, weights=qcd['EventWeight'], label='QCD', color='salmon')
      plt.xlabel(key)
      plt.yscale('log')
      plt.legend(loc='best')
      #plt.show()
      print('Saving:',folder + '/' + str(key) + '.png')
      plt.savefig(folder + '/' + str(key)+'.png')

# This will only contain TTbar, Drell Yan and W+jets events (all branches). You are concatenating them.
df_full = pd.concat((ttbar, dy, wj), ignore_index=True)
# However, we decided we were only going to train on event-level variables, so this is would be a more useful df:
df =  pd.concat((ttbar[npart], dy[npart], wj[npart]), ignore_index=True)
if debug:
  print('----> My data frame:')
  print(df)

# Turn Data into ML-Compatible Inputs
# Keras, just like scikit-learn, takes as inputs the following objects: 
# 1) X: an ndarray of dimensions [nb_examples, nb_features] containing the distributions to be used as inputs to the model. Each row is an object to classify, each column corresponds to a specific variable.
# 2) Y: an array of dimensions [nb_examples] containing the truth labels indicating the class each object belongs to (for classification), or the continuous target values (for regression).
# 3) (optional) w:an array of dimensions [nb_examples] containing the weights to be assigned to each example

# Turn the df the desired ndarray "X" that can be directly used for ML applications.
X = df.as_matrix() # Each row is an object to classify, each column corresponds to a specific variable.
# Take the weights
w =  pd.concat((ttbar['EventWeight'], dy['EventWeight'], wj['EventWeight']), ignore_index=True).values
# This is the array with the true values
y = []
for _df, ID in [(ttbar, 0), (dy, 1), (wj, 2)]:
    y.extend([ID] * _df.shape[0])
y = np.array(y)

# Randomly shuffle and automatically split all your objects into train and test subsets
ix = range(X.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks
X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=0.6) # Train here is 60% of the statistic
# It is common practice to scale the inputs to Neural Nets such that they have approximately similar ranges.
# Without this step, you might end up with variables whose values span very different orders of magnitude.
# This will create problems in the NN convergence due to very wild fluctuations in the magnitude of the internal weights.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Multilayer Perceptron (MLP) definition
model = Sequential()
# Dense: Core layer of an MLP. input_dim necessary arguments for the 1st layer of the net. Activation 'relu' is suggested for the first layer.
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu')) # Linear transformation of the input vector. The first number is output_dim.
model.add(Dropout(0.2)) # To avoid overfitting. It masks the outputs of the previous layer such that some of them will randomly become inactive and will not contribute to information propagation.
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Highway(activation='relu')) # Use adaptive gating units which learn to regulate the flow of information through a network. Improves ability to train very deep feed-forward nets.
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(MaxoutDense(5, 4)) # A Dense layer that learns its own activation function.
model.add(Dense(3, activation='softmax')) # Last layer has to have the same dimensionality as the number of classes we want to predict, here 3.
# Summary of your DNN.
model.summary()
# Now you need to declare what loss function and optimizer to use (and compile your model).
model.compile('adam', 'sparse_categorical_crossentropy')
print('---------------------------Training:---------------------------')
try:
    model.fit(X_train, y_train, class_weight={
                0 : 0.33 * (float(len(y)) / (y == 0).sum()), # (y == 0).sum() give the total number of elemnet that are zero in y (array of the true category).
                1 : 0.33 * (float(len(y)) / (y == 1).sum()),
                2 : 0.33 * (float(len(y)) / (y == 2).sum())
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
# Turn them into classes
yhat_cls = np.argmax(yhat, axis=1) # Returns the indices of the maximum values along an axis
bins = np.linspace(0,3,4)
fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
_ = plt.hist(yhat_cls, bins=bins, histtype='stepfilled', alpha=0.5, label='prediction', weights=w_test)
_ = plt.hist(y_test, bins=bins, histtype='stepfilled', alpha=0.5, label='truth', weights=w_test)
plt.legend(loc='upper left')
#plt.show()
print('Saving:',folder + '/Performance.png')
plt.savefig(folder + '/Performance.png')

# Signal eff = weighted tpr --> out of all signal events, what % for we classify as signal?
print 'Signal efficiency:', w_test[(y_test == 0) & (yhat_cls == 0)].sum() / w_test[y_test == 0].sum()

# Bkg eff = weighted fpr --> out of all bkg events, what % do we classify as signal?
b_eff = w_test[(y_test != 0) & (yhat_cls == 0)].sum() / w_test[y_test != 0].sum()
print 'Background efficiency:', b_eff
print 'Background rej:', 1. / b_eff
# Events that got assigned to class 0
predicted_ttbar = df_full.ix[np.array(ix_test)[np.argmax(yhat, axis=1) == 0]]
# Where all ttbar examples assigned to class 0?
for classID in np.unique(y_test):
    plt.hist(np.argmax(yhat[y_test == classID], axis=1), weights=w_test[y_test == classID])
    print('Saving:',folder + '/ttbarInclass0_' + str(classID) + '.png')
    plt.savefig(folder + '/ttbarInclass0_' + str(classID) + '.png')
    #plt.show()
print("Sum:",sum(w_test[(np.argmax(yhat, axis=1) !=0) & (y_test == 0)]))

# Recurrent Neural Networks
# Let's look at a fancier way of solving the same classification problem. In this case we will use Recurrent Neural Netwroks. These allow you to process variable length sequences of data.
# For example, we can use them to describe an event in terms of the properties of its jets, whose number varies event by event.
# We could also describe the same event using the properties of its muons, or any other particle that appears in it.
# Because the number of particles of each type changes in each event, we need the flexibility of RNNs to process this type of data.
jetvars = [key for key in df_full.keys() if key.startswith('Jet')]
jetvars.remove('Jet_ID')
print jetvars
muonvars = [key for key in df_full.keys() if key.startswith('Muon')]
print muonvars
photonvars = [key for key in df_full.keys() if key.startswith('Photon')]
print photonvars
electronvars = [key for key in df_full.keys() if key.startswith('Electron')]
print electronvars
df_jets = df_full[jetvars].copy()
df_electrons = df_full[electronvars].copy()
df_muons = df_full[muonvars].copy()
df_photons = df_full[photonvars].copy()

num_electrons = max([len(e) for e in df_electrons.Electron_E])
num_muons = max([len(m) for m in df_muons.Muon_E])
num_photons = max([len(gam) for gam in df_photons.Photon_E])
num_jets = max([len(j) for j in df_jets.Jet_E])

# Since I'm not going to retrain this net right now but I'm just loading in some pre-trained weights, I will also need to load the exact ordering of indices that I used the time I trained.
# This way, I will ensure that the random shuffling in train_test_split will not cause me to evaluate on a subset of examples I previously trained on.
ix_train, ix_test = cPickle.load(open('./models/ixtraintest.pkl', 'rb'))
X_train, y_train, w_train = X[ix_train], y[ix_train], w[ix_train]
X_test, y_test, w_test = X[ix_test], y[ix_test], w[ix_test]
# Just for the sake of variety, you can either have class labels like (0, 1, 2, 3, ...) and train using spare_categorical_crossentropy as you loss function -- like we did before -- or, equivalently,
# you can have class labels like ([1, 0, 0, 0, ...], [0, 1, 0, 0, ...], ...) and train using categorical_crossentropy.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

Xjet_train, Xjet_test = create_stream(df_jets, num_jets, sort_col='Jet_btag')
Xphoton_train, Xphoton_test = create_stream(df_photons, num_photons, sort_col='Photon_E')
Xmuon_train, Xmuon_test = create_stream(df_muons, num_muons, sort_col='Muon_E')
Xelectron_train, Xelectron_test = create_stream(df_electrons, num_electrons, sort_col='Electron_E')
jet_channel = Sequential()
muon_channel = Sequential()
electron_channel = Sequential()
photon_channel = Sequential()
JET_SHAPE = Xjet_train.shape[1:]
MUON_SHAPE = Xmuon_train.shape[1:]
ELECTRON_SHAPE = Xelectron_train.shape[1:]
PHOTON_SHAPE = Xphoton_train.shape[1:]
jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
jet_channel.add(GRU(25, name='jet_gru'))
jet_channel.add(Dropout(0.3, name='jet_dropout'))

muon_channel.add(Masking(mask_value=-999, input_shape=MUON_SHAPE, name='muon_masking'))
muon_channel.add(GRU(10, name='muon_gru'))
muon_channel.add(Dropout(0.3, name='muon_dropout'))

electron_channel.add(Masking(mask_value=-999, input_shape=ELECTRON_SHAPE, name='electron_masking'))
electron_channel.add(GRU(10, name='electron_gru'))
electron_channel.add(Dropout(0.3, name='electron_dropout'))

photon_channel.add(Masking(mask_value=-999, input_shape=PHOTON_SHAPE, name='photon_masking'))
photon_channel.add(GRU(10, name='photon_gru'))
photon_channel.add(Dropout(0.3, name='photon_dropout'))
combined_rnn = Sequential()

combined_rnn.add(Merge([
            jet_channel, muon_channel, electron_channel, photon_channel
        ], mode='concat'))

combined_rnn.add(Dense(64, activation = 'relu'))
combined_rnn.add(Dropout(0.3))
combined_rnn.add(Highway(activation = 'relu'))
combined_rnn.add(Dropout(0.3))
combined_rnn.add(Highway(activation = 'relu'))
combined_rnn.add(Dropout(0.3))
combined_rnn.add(Dense(3, activation='softmax'))
combined_rnn.summary()
combined_rnn.compile('adam', 'categorical_crossentropy')
print 'Training RNN:'
try:
    combined_rnn.fit([Xjet_train, Xmuon_train, Xelectron_train, Xphoton_train], y_train, batch_size=16,
            class_weight={
                0 : 0.33 * (float(len(y)) / (y == 0).sum()),
                1 : 0.33 * (float(len(y)) / (y == 1).sum()),
                2 : 0.33 * (float(len(y)) / (y == 2).sum())
        },
        callbacks = [
            EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
            ModelCheckpoint('./models/combinedrnn_tutorial-progress', monitor='val_loss', verbose=True, save_best_only=True)
        ],
    nb_epoch=30, 
    validation_split = 0.2) 

except KeyboardInterrupt:
    print 'RNN training ended early.'
# -- load in best network
combined_rnn.load_weights('./models/combinedrnn_tutorial-progress')
print 'Saving RNN weights...'
combined_rnn.save_weights('./models/combinedrnn_tutorial.h5', overwrite=True)

json_string = combined_rnn.to_json()
open('./models/combinedrnn_tutorial.json', 'w').write(json_string)
yhat_rnn = combined_rnn.predict([Xjet_test, Xmuon_test, Xelectron_test, Xphoton_test], verbose = True, batch_size = 512)
bins = np.linspace(0,3,4)
fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
_ = plt.hist(np.argmax(yhat_rnn, axis=1), bins=bins, histtype='stepfilled', alpha=0.5, label='prediction', weights=w_test)
_ = plt.hist(y_test.argmax(axis=1), bins=bins, histtype='stepfilled', alpha=0.5, label='truth', weights=w_test)
plt.legend(loc='upper left')
#plt.show()
print('Saving:',folder + '/PerformanceRNN.png')
plt.savefig(folder + '/PerformanceRNN.png')
# -- turn the predictions back into class labels
yhat_rnn_cls = np.argmax(yhat_rnn, axis=1)
yhat_rnn_cls
# -- do the same for the truth labels
y_test_cls = np.argmax(y_test, axis=1)
print 'Signal efficiency:', w_test[(y_test_cls == 0) & (yhat_rnn_cls == 0)].sum() / w_test[y_test_cls == 0].sum()
b_eff = w_test[(y_test_cls != 0) & (yhat_rnn_cls == 0)].sum() / w_test[y_test_cls != 0].sum()
print 'Background efficiency:', b_eff
print 'Background rej:', 1 / b_eff

