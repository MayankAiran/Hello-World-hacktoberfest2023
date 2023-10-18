from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Define the model
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(None, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size=32)
