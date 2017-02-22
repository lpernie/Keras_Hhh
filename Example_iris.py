# http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# In this example we train a DNN to make categorization. We have 4 variables describing a flower (features) and 3 flower names (class values).
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, MaxoutDense, Highway
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from keras.models import Sequential
debug = True
folder="./files/"
# Fix random seed for reproducibility.
seed = 7
numpy.random.seed(seed)
# Load dataset containing features and the class values.
dataframe = pandas.read_csv(folder+"iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float) # the first 4 floats, that are the features.
Y = dataset[:,4]                 # the last item, that is the class values (3 different folwer names).
if debug:
  print("X is: ------------ "); print(X) # (list of array of 4 floats).
  print("Y is: ------------ "); print(Y) # (list of strings, i.e. the class values).
# Encode class values as integers. They are initially strings, now Iris-virginica=1, Iris-versicolor=2, etc...
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
if debug:
  print("Encoded Y is: ------------ "); print(encoded_Y)
# Instead than 1,2, and 3 now there is an array of 3. If there is the first class values you have (1,0,0), if the second (0,1,0)... This is the format the DNN can use.
dummy_y = np_utils.to_categorical(encoded_Y)
if debug:
  print("Dummy Y is: ------------ "); print(dummy_y)
# Define baseline model you will call.
def baseline_model():
  model = Sequential() # standard inizialization
  # We use this schema: 4 inputs -> [4 hidden nodes] -> 3 outputs
  model.add(Dense(4, input_dim=4, init='normal', activation='relu')) # add a layer. Output is size 4 (the features), input too. Relu is a good activation function to start.
  model.add(Highway(activation='relu')) # improve the DNN by going fast where is not important, and slower where it matters.
  model.add(Dropout(0.05)) # this remove X connection each time, so that you do not overtrain
  model.add(MaxoutDense(4,4)) # it find it own activation function. It takes output dim and feature as arguments.
  model.add(Dense(3, init='normal', activation='sigmoid'))           # another laayer. 3 final outputs, the possible class values. Sigmoid is a good activation function to finish.
  #print("The model you have is:");
  #model.summary()
  # Compile model. You need to provide the optimizer, the loss function and the metrics. We use ADAM gradient descent optimization algorithm with a loga loss function.
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
# KerasClassifier takes the name of a function as an argument. This function must return the constructed neural network model, ready for training (compiled).
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
# You want to train the DNN on 67% of the data and test in the remaining 33%.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
# The real training is not done only with X_train and Y_train
estimator.fit(X_train, Y_train)
# And you can make now a prediction on the X_test
predictions = estimator.predict(X_test)
print("Prediction (in integers): ------------ ")
print(predictions)
print("Prediction (translated to the initial strings): ------------ ")
print(encoder.inverse_transform(predictions))
# Check accuracy. We set the number of folds to be 10 (excellent default) and to shuffle the data before partitioning it.
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("-> Accuracy is: %.2f%% with %.2f%% uncertainty." % (results.mean()*100, results.std()*100))

# Save model to JSON (JSON is a file format for describing data hierarchically).
model_json = estimator.model.to_json()
with open(folder+"model.json", "w") as json_file:
    json_file.write(model_json)
# Save weights to HDF5 formats
estimator.model.save_weights(folder+"model.h5")

# Later you can access the model and the weights again.
# Load json and create model
json_file = open(folder+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights(folder+"model.h5")
# Now you have to compile the model again 
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# We now evaluate the performance. I use anothe method (no Kfold), and I simply see how many time I got the correct value on all dataset
score = loaded_model.evaluate(X, dummy_y, verbose=0)
print("Loaded model has accuracy: %.2f%%" % (score[1]*100))
# I can do the same but only on the data part not used in training
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("Loaded model has accuracy: %.2f%%, found on test sample only" % (score[1]*100))
score = loaded_model.evaluate(X_train, Y_train, verbose=0)
print("Loaded model has accuracy: %.2f%%, found on train sample only" % (score[1]*100))
