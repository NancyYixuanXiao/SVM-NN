'''
The following script performs 5-fold cross-validation to train and evaluate a neural
network classifier.  The structure of the network consists 4 layers including the (i) 30x1 input layer,
(ii) 50 x 1 hidden layer, (iii) 20x1 hidden layer, and (iv) a 1x1 output layer.  The network uses a
cross-entropy cost function to speed up training and a verison of stochatic gradient descent
called "adam".  The network also uses a regularization technique called Dropout to reduce
overfitting.

The script using Keras deep learning software framework to build and update the network.
An Amazon GPU (instance type: g2.2xlarge) was used to reduce network training times.
'''

# Load required classes and functions
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2, activity_l2
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load Wisconsin Diagnostic Breast Cancer Dataset
dataset = numpy.loadtxt("wpbc_24_month_threshold.csv", delimiter=",")

# Create input features and output classification data
X = dataset[:,0:32]
Y = dataset[:,32]

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(50, input_dim=32, init='normal', activation='relu'))
	model.add(Dropout(0.40))
	model.add(Dense(20, init='normal', activation='relu'))
	model.add(Dropout(0.40))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=500, batch_size=150, verbose=2)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
