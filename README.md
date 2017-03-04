# Description
The release contains 3 folders containing examples on training DNN with keras:  
   1. Examples_Generic (generic keras examples);  
   2. Examples_Ntuples (keras example from real root Ntuples);  
   3. Hhh_analysis (real Keras code got Hhh analysis);  

# Notes  
If you are on lxplus, you need to be in release "CMSSW_9_0_X_2017-02-27-2300" or older to have keras installed.  
At the moment tensorflow is not installed in any release (you have to use theano). You can switch to theano by:  
```
vim ~/.keras/keras.json
:%s/tensorflow/theano/g
```

# Intructions:  
In order to run Keras on the tt and Hhh ntuples:  
```
git clone https://github.com/lpernie/Keras_Hhh.git;  
cd Keras_Hhh/Hhh_analysis/files_root;  
source GetTheFiles.sh;  
cd ../;  
python Hhh_tt_MLP.py;
```

# DNN with Keras
   1. Usually you train with 50% of signal and backgrund. This is done by using the same numbers of events in the 2 classes, or giving a weight to the classes to make them count the same. You can set the weights with class_weight (single weights for each events of a class), and sample_weight (array of weights for each event in a class) and give them to the funciton "model.fit()".   
   2. There is no special receipt for the best MVA. Usually you can try to play with the weights of the initialization, the parameters of the optimizer, epoch, the activation functions, the validation spits ecc.   
   3. It can be useful to rank the parameter for how they are useful. While in a BTD this is easy, for NN is not possible. You can make a a-priori ranking, before cretaing the classifier, but this assume that is a linear model, so it has to be taken cum-grano-salis. Details in "https://github.com/mickypaganini/bbyy_jet_classifier/blob/master/bbyy_jet_classifier/process_data.py#L113"   
   4. Correlation among features can be estimated using the Pierson coefficient
   5. To perform a parametric training you need an extra input to the neural network: the mass. You have to do the training passing events coming from different samples (different mass), passing the mass value as a input of the NN.   
