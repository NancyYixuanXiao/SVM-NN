'''
The following script performs a grid search on the normalized WDBC dataset to determine the optimal size for a
neural network with two hidden layers.  The optimal size is determine by the neural
network with the maximum classifcation accuracy (or sensitivity).  Layers sizes between 30 and 90 neurons
in increments of 20 are tested for layer 1.  Layers sizes between 10 and 30 are neurons
in increments of 10 are tested for layer 2.  5-fold cross validation is used for each
iteration of grid search.  Additionally, the dropout regularization technique is in each iteration.

The script using Keras deep learning software framework to build and update the network.
An Amazon GPU (instance type: g2.2xlarge) was used to reduce network training times.
'''

# Load required classes and functions from Keras and Numpy
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
from sklearn.model_selection import GridSearchCV

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load the min-max normalized Wisconsin Diagnostic Breast Cancer Dataset
dataset = numpy.loadtxt("WDBC_norm.csv", delimiter=",")

# Create input features and output classification data
X = dataset[:,0:30]
Y = dataset[:,30]

# Below function builds the neural network structure with the appropriate layer sizes
def create_baseline(layer1 = 1, layer2 = 1):
	# create model
	model = Sequential()
	model.add(Dense(layer1, input_dim=30, init='normal', activation='relu'))
	# model.add(Dropout(0.40))
	model.add(Dense(layer2, init='normal', activation='relu'))
	# model.add(Dropout(0.40))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# Build neural network with required parameters
model = KerasClassifier(build_fn=create_baseline, nb_epoch=500, batch_size=100, verbose=2)

# Perform grid search to determine optimal layer sizes
layer1 = [10, 30, 50, 70, 90, 110]
layer2 = [10, 20, 30, 40]
param_grid = dict(layer1 = layer1, layer2 = layer2)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator = model,param_grid = param_grid, cv=kfold, scoring = 'accuracy')
grid_result = grid.fit(X,Y)

# Determine opitmal layer size and print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Save results to be used for visualization
wrt = open("results.txt", 'w')
wrt.write('layer1' + ',' + 'layer2' + ',' + 'mean\n')
for i in range(len(params)):
    wrt.write(str(params[i]['layer1'])+','+str(params[i]['layer2'])+','+str(means[i])+'\n')
wrt.close()
