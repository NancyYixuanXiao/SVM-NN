'''
The following script performs a grid search on the normalized WDBC dataset to determine the optimal
amount of regularization for the neural network.  The grid search will assess pairs of two features
to determine the optimal degree of regularization: dropout percentage and maximum weight constraints.
Dropout percentage refers to the percentage of neurons in the hidden layers that are removed in each
training iteration.  Maximum weight limits puts a cap on the weighted connections between all of the neurons.
The optimal regularization is determine by the neural network with the maximum classifcation accuracy.
Dropout percentages between 0 and 90 percent in increments of 10 and max weights constraints between 1 and 5 in increments of 1
are tested.  Additionally, the grid seach will use the optimal neural network size determined in the
WDBC_norm_GS_Layer_Size script (Layer 1: 30 neurons, Layer 2: 40 neurons).

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
from keras.constraints import maxnorm

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load the min-max normalized Wisconsin Diagnostic Breast Cancer Dataset
dataset = numpy.loadtxt("WDBC_norm.csv", delimiter=",")

# Create input features and output classification data
X = dataset[:,0:30]
Y = dataset[:,30]

# Below function builds the neural network structure with the appropriate regularization
def create_baseline(dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(90, input_dim=30, init='normal', activation='relu', W_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(30, init='normal', activation='relu', W_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# Build neural network with required parameters
model = KerasClassifier(build_fn=create_baseline, nb_epoch=500, batch_size=100, verbose=2)

# Perform grid search to determine optimal degree of regularization
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator = model,param_grid = param_grid, cv=kfold, scoring = 'recall')
grid_result = grid.fit(X,Y)

# Determine opitmal regularization and print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Save results to be used for visualization
wrt = open("results.txt", 'w')
wrt.write('weight_constraint' + ',' + 'dropout_rate' + ',' + 'mean\n')
for i in range(len(params)):
    wrt.write(str(params[i]['weight_constraint'])+','+str(params[i]['dropout_rate'])+','+str(means[i])+'\n')
wrt.close()
