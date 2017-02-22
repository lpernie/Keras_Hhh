# Description
The release contains 3 folders containing examples on training DNN with keras:  
   1. Examples_Generic (generic keras examples);  
   2. Examples_Ntuples (keras example from real root Ntuples);  
   3. Hhh_analysis (real Keras code got Hhh analysis);  

# Notes  
If you are on lxplus, you need to be in release "CMSSW_9_0_X_2017-02-14-1100" or older to have keras installed.  
At the moment tensorflow is not installed in any release (you have to use theano). You can switch to theano by:  
```
vim ~/.keras/keras.json
:%s/tensorflow/theano/g
```

# Intructions:  
In order to run Keras on the tt and Hhh ntuples:  
```
git clone https://github.com/lpernie/Keras_Hhh.git;  
cd Hhh_analysis/files_root;  
source GetTheFiles.sh;  
cd ../;  
python Hhh_tt_MLP.py;
```
