"""parse json data"""

import json
import numpy as np
import neural
def unique(list1):
    x = np.array(list1)
    y=np.unique(x)
    return y



def main():
    data = {}
    ids = []
    ingredients = []
    actualing=[]
    cuisine_types = []
    with open("train.json", "r") as read_file:
        data = json.load(read_file)
    for recipe in data:
        ids.append(recipe['id'])
        ingredients.extend(recipe['ingredients'])
        actualing.append(recipe['ingredients'])
        cuisine_types.append(recipe['cuisine'])
    #print("data shapes")
    #print("ingredients:", ingredients.shape)
    #print("ids:", ids.shape)
    #print("cuisine_types:", cuisine_types)
    y=unique(ingredients)
    g=unique(cuisine_types)
    convdata=[]
    gg=list(g)
    for i in range(len(actualing)):
        x=createOneHot(y,actualing[i])
        index=gg.index(cuisine_types[i])
        list3=[]
        list3.append(cuisine_types[i])
        cusineoh=createOneHot(gg,list3)
        convdata.append((x,cusineoh))

    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

    # Construct the model
    # The first layer must provide the input shape
    neural_net = Sequential()
    neural_net.add(Dense(6714, activation='relu'))
    neural_net.add(Dense(10, activation='softmax'))
    #neural_net.summary()
    x_train_vectors=[]
    y_train_vectors=[]
    x_test_vectors=[]
    y_test_vectors=[]
    i=0
    for element in convdata:
        if i<20000:
            x_train_vectors.append(element[0])
            y_train_vectors.append(element[1])
        else:
            x_test_vectors.append(element[0])
            y_test_vectors.append(element[1])
        i=i+1
    # Compile the model
    time.sleep(5)
    neural_net.compile(optimizer="SGD", loss="categorical_crossentropy",
                       metrics=['accuracy'])
    # Train the model
    i=0
    for i in range(100):
        i=i+1
        print("hello")
    history = neural_net.fit(x_train_vectors, y_train_vectors, verbose=0,
                             validation_data=(x_test_vectors, y_test_vectors),
                             epochs=5)
    i=0
    for i in range(100):
        i=i+1
        print("hello")
    loss, accuracy = neural_net.evaluate(x_test_vectors, y_test_vectors, verbose=0)
    print("accuracy: {}%".format(accuracy*100))

    # Examine which test data the network is failing to predict
    import matplotlib.pyplot as plt
    from numpy import argmax
    from numpy.random import randint
    """
    outputs = neural_net.predict(x_test_vectors)
    answers = [argmax(output) for output in outputs]
    targets = [argmax(target) for target in y_test_vectors]
    """
    """
    for i in range(len(answers)):
        if answers[i] != targets[i]:
            print("Network predicted", answers[i], "Target is", targets[i])
            plt.imshow(x_test[i], cmap='gray')
            plt.show()
    """
    """
    for input_vector, target_vector in convdata:
        output_vector = nn.predict(input_vector)
        #print("target:", target_vector, "output:", output_vector)
    """
    print(accuracy)
main()
