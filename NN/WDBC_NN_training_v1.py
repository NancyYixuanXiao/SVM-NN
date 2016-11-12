from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load Wisconsin Diagnostic Breast Cancer Dataset
dataset = numpy.loadtxt("WDBC_processed.csv", delimiter=",")
# testset = numpy.loadtxt("happy-sad-privtest-1.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:30]
Y = dataset[:,30]

# X_test = testset[:,0:2304]
# Y_test = testset[:,2304]

# create model
model = Sequential()
model.add(Dense(8, input_dim=30, init='normal', activation='relu'))
# model.add(Dense(8, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch = 500, batch_size = 100)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# evaluate test set
# loss, accuracy = model.evaluate(X_test, Y_test)
# print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
