# Import required modules
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from keras import Sequential
from keras.layers import Dense

# Import training and testing data
training = pd.read_csv('../../Datasets/train.csv')
testing = pd.read_csv('../../Datasets/test.csv')


# Obtaining feature and target values of training data
train_features = []
for i in range(300):
	train_features.append(str(i))
xtrain = training[train_features]
ytrain = training['target']
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)

# Obtaining feature values of testing data
test_features = []
for i in range(300):
	test_features.append(str(i))
xtest = testing[test_features]
sc = StandardScaler()
xtest = sc.fit_transform(xtest)

# Define ANN Model
classifier = Sequential()
classifier.add(Dense(4, activation = 'relu', kernel_initializer = 'random_normal', input_dim = 300))
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal'))

# Compile the ANN Model
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN Model to test data
classifier.fit(xtrain,ytrain, batch_size = 5, epochs = 250)

# Test the model
ytest = classifier.predict(xtest)
ytest = [1 if i > 0.60 else 0 for i in ytest]

# Export result values to a csv file to test on kaggle
id = []
for i in range(250, 20000):
	id.append(i)
result = {'id': id, 'target': ytest}
result = pd.DataFrame(result)
result.to_csv("result.csv", index = False)
