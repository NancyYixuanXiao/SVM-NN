# Load required classes and functions
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
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
dataset = numpy.loadtxt("WDBC_processed.csv", delimiter=",")

# Create input features and output classification data
X = dataset[:,0:30]
Y = dataset[:,30]

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(50, input_dim=30, init='normal', activation='relu'))
        # model.add(Dense(10, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=500, batch_size=100, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
