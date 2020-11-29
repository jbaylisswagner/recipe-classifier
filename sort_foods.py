"""parse json data"""

import json
import numpy as np
import neural
import csv
from numpy import argmax

def processIngredients(ingredientsMap, total_ingredients_num, recipe):
    """
    inputs:
    -dict mapping ingredients to distinct integers
    -int number of all unique ingredients
    -recipe: list of ingredients (strings) in the recipe
    purpose: make onehot vector the length of the
    number of possible ingredients where activation
    is 1 for any ingredients present in the recipe
    return: onehot vector in form of numpy array
    """
    ing_array = np.zeros(total_ingredients_num)
    for ingredient in recipe:
      ing_num = ingredientsMap[ingredient]
      ing_array[ing_num] = 1
    #print("ingredients: ", recipe)
    #print("numpy array onehot thing", ing_array)
    return ing_array
"""
ingredients: list of unique ingredients
recipe: ingredients in the given recipe
"""
def createOneHot(all_ingredients,recipe):
    """
    create multiple hot vector where 1s represent
    presence of an ingredient in a recipe, because
    each index represents a unique ingredient
    """
    onehot=[0]*len(all_ingredients) #creates a list of zeros of length 6714
    listlist=list(all_ingredients)
    for element in recipe:
        index=listlist.index(element)
        onehot[index]=1
    return onehot

def unique(list1):
    """
    input: list
    return: list without duplicates (so only 'unique' items)
    """
    newlist = np.array(list1)
    num_unique = np.unique(newlist)
    return num_unique

def map_data(ingredients,outputs, cuisine_map, ing_map):
    """
    input:
    -list of all ingredients
    used in training and testing data
    -list of every output (all cuisines)
    -dictionary to map
    ingredients to numerical (int) label
    -dict to map cuisine types to label

    purpose: fill dictionary will all possible
    items so that they all have labels
    """
    unique_ingredients_lst = list(unique(ingredients))
    num_ingredients = len(unique_ingredients_lst)

    #change numpy array into list
    #cuisines lst will have every possible different type of cuisines w no repeat
    unique_cuisines_lst = list(unique(outputs))
    #print("unique cuisines list: ", unique_cuisines_lst)
    num_cuisines = len(unique_cuisines_lst)

    for i in range(num_cuisines):
        #print("cuisine =", unique_cuisines_lst[i], "; num label = ", i)
        cuisine_map[unique_cuisines_lst[i]] = i


    for i in range(num_ingredients):
        ing_map[unique_ingredients_lst[i]] = i


def process_data_files(filename, recipes,ids, outputs, ingredients):
    """
    inputs:
    -filename: string filename for dataset
    -ids: list of labels for data points
    -recipes: list of lists of ingredients. X data
    -outputs: cuisine types. Y data
    -ingredients: list of all ingredients that trainigng & testing data mention

    purpose: modify all of these lists using info from the data set
    to create parallel lists
    """

    data = {}

    with open(filename, "r") as read_file:
        data = json.load(read_file)
    for recipe in data:
        ids.append(recipe['id'])
        #adds each ingredient individually
        ingredients.extend(recipe['ingredients'])
        #recipes: input list
        recipes.append(recipe['ingredients'])
        #cuisine types: output list
        #TODO: make sure this doesn't crash for data w/o cuisine type
        if filename == "train.json":
            outputs.append(recipe['cuisine'])

def main():

    labels_train_X = []
    train_X = []
    ingredients = []
    train_Y = []

    process_data_files("train.json",train_X, labels_train_X, train_Y, ingredients)

    test_X = []
    labels_test_X = []
    test_Y = []

    process_data_files("test.json", test_X, labels_test_X, test_Y, ingredients)


    cuisine_map = {}
    #create dictionaries mapping cuisine and ingredient types to a unique number
    ing_map = {}
    rcmap={}
    map_data(ingredients, train_Y, cuisine_map, ing_map)

    #for each recipe, create a one hot vector to tell us its ingredients
    #makes all cuisine types into different number


    train_X_vectors = []
    train_Y_vectors = []
    test_X_vectors = []

    #make training and testing data into vectors
    num_ingredients = len(list(unique(ingredients)))
    for i in range(len(train_X)):
        curIngredients = train_X[i]
        curCuisine = train_Y[i]
        cuisineNum = cuisine_map[curCuisine]
        rcmap[cuisineNum]=curCuisine
        ingredientsVector = processIngredients(ing_map, num_ingredients, curIngredients)
        train_X_vectors.append(ingredientsVector)
        #output for this data point is its cuisine type
        train_Y_vectors.append(cuisineNum)

    for i in range(len(test_X)):
        curIngredients = test_X[i]
        ingredientsVector = processIngredients(ing_map, num_ingredients, curIngredients)
        test_X_vectors.append(ingredientsVector)

    from keras.utils import to_categorical
    train_Y_vectors = to_categorical(train_Y_vectors,len(cuisine_map))
    #show us what data looks like

    #ml stuff now
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

    output_shape = train_Y_vectors.shape
    print("output shape: ", output_shape)

    train_X_vectors = np.array(train_X_vectors)
    test_X_vectors = np.array(test_X_vectors)
    input_shape = train_X_vectors.shape
    print("input shape: ", input_shape)


    # Construct the model
    # The first layer must provide the input shape

    neural_net = Sequential()
    neural_net.add(Dense(3000, input_shape = (7137, ), activation='relu'))
    #neural_net.add(Dense(1000, activation = 'relu'))
    neural_net.add(Dense(500, activation = 'relu'))
    neural_net.add(Dense(20, activation='softmax'))
    neural_net.summary()

    neural_net.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=['accuracy'])

    history = neural_net.fit(train_X_vectors, train_Y_vectors, verbose=1, epochs = 15)

    #loss, accuracy = neural_net.evaluate(test_X, test_Y, verbose=0)
    #print("accuracy: {}%".format(accuracy*100))
    print(test_X_vectors)

    test_Y= neural_net.predict(test_X_vectors)
    print(test_Y[i])
    with open('out.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        row=[]
        for i in range(len(test_Y)):
            row.append([labels_test_X[i],rcmap[argmax(test_Y[i])]])
        writer.writerows(row)

    csvFile.close()


main()
