



# Determine the range of the data (max and min values)
from keras.utils import to_categorical
x_max = x_train.max()
x_min = x_train.min()

print("-"*60)
print("x_max is", x_max)
print("x_min is", x_min)

# Determine the number of output classes
num_categories = len(set(y_train))
print("num_categories in output", num_categories)
# Normalize pixel values to range [0,1]
x_train_vectors = (x_train.reshape(60000,28,28,1)-x_min) / float(x_max - x_min)
x_test_vectors = (x_test.reshape(10000,28,28,1)-x_min) / float(x_max - x_min)
y_train_vectors = to_categorical(y_train, num_categories)
y_test_vectors = to_categorical(y_test, num_categories)

print("-"*60)
print("data shapes")
print("  training input:", x_train_vectors.shape)
print("  testing input:", x_test_vectors.shape)
print("  training output:", y_train_vectors.shape)
print("  testing output:", y_test_vectors.shape)

"""
# Verify that the training data is the format you expect
print("input array example:\n", x_train_vectors[0])
print("one-hot encoding example:\n", y_train_vectors[0])
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Construct the model
# The first layer must provide the input shape
neural_net = Sequential()
neural_net.add(Conv2D(16,(5,5),activation="relu",input_shape=(28,28,1)))
neural_net.add(MaxPooling2D(pool_size=(2,2)))
neural_net.add(Conv2D(32,(5,5),activation="relu",input_shape=(28,28,1)))
neural_net.add(MaxPooling2D(pool_size=(2,2)))
neural_net.add(Flatten())
neural_net.add(Dense(100, activation='relu'))
neural_net.add(Dense(10, activation='softmax'))
neural_net.summary()

# Compile the model
neural_net.compile(optimizer="SGD", loss="categorical_crossentropy",
                   metrics=['accuracy'])

# Train the model
history = neural_net.fit(x_train_vectors, y_train_vectors, verbose=1,
                         validation_data=(x_test_vectors, y_test_vectors),
                         epochs=5)

loss, accuracy = neural_net.evaluate(x_test_vectors, y_test_vectors, verbose=0)
print("accuracy: {}%".format(accuracy*100))

# Examine which test data the network is failing to predict
import matplotlib.pyplot as plt
from numpy import argmax
from numpy.random import randint

outputs = neural_net.predict(x_test_vectors)
answers = [argmax(output) for output in outputs]
targets = [argmax(target) for target in y_test_vectors]

for i in range(len(answers)):
    if answers[i] != targets[i]:
        print("Network predicted", answers[i], "Target is", targets[i])
        plt.imshow(x_test[i], cmap='gray')
        plt.show()
