import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD,Adam

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
df = pd.read_csv("sample_data/bank-full.csv", sep=";")

Y = df.values[:,16]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Y_encoded = np_utils.to_categorical(encoded_Y)

X = df.drop('y', 1)

print (X)
print (Y_encoded)


# one hot encoding
categorical_features_names = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in categorical_features_names:
	one_hot = pd.get_dummies(X[col])
	one_hot = one_hot.add_prefix(col)
	X = X.join(one_hot)  # adding onehot encoded columns to dataframe
	X = X.drop(col, 1)  # deleting categorical column


# number of inputs to the first layer
n_cols = X.shape[1]

X = X.values
print(type(X))
print(type(Y_encoded))


model = Sequential()
# model.add(Dense(50, activation='tanh', input_shape=(n_cols,)))
# model.add(Dense(2, activation='sigmoid'))

model.add(Dense(128, activation='relu', input_shape=(n_cols,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# determining optimizer, loss, and metrics:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model training
model.fit(X, Y_encoded)

