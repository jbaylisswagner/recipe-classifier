from numpy import dot, exp, allclose,array
from numpy.random import normal, shuffle, seed

from functools import partial
from itertools import product


class Node(object):
    def __init__(self, name, learning_rate=1.0):
        self.name = name
        self.in_edges = []
        self.out_edges = []
        self.activation = 0.0 # set by self.compute_activation
        self.delta = 0.0 # set by self.compute_*_delta
        self.learning_rate = learning_rate

    def __repr__(self):
        return self.name

class InputNode(Node):
    """A node whose only purpose is to store an activation.

    The first layer of a neural network will consist of InputNode objects.
    Input nodes should have no in_edges, and will never update self.delta.
    They have no activation function, and instead get their activation set
    directly by the neural network."""
    def __init__(self, name):
        Node.__init__(self, name, 0.0)

class BiasNode(Node):
    """A node that always stores activation=1.

    A neural network has a single bias node that does not appear in any layer.
    However, the bias node is connected to all hidden and output nodes, and
    the weights on the edges from bias to those nodes will be updated during
    backpropagation."""
    def __init__(self):
        Node.__init__(self, "bias")
        self.activation = 1.0

class SigmoidNode(Node):
    """Neuron with a sigmoid activation function.
    All methods update internal state, and therefore have no return value."""
    def compute_activation(self):
        """Computes 1 / (1 + e^-x) where x is the weighted sum of inputs.

        Stores the result in self.activation. Assumes that activations
        for nodes in the previous layer are already up to date."""

        x=0
        edges=self.in_edges
        for edge in edges:
            x+= edge.source.activation*edge.weight
        self.activation=1/(1+exp(-x))


    def compute_output_delta(self, target):
        """Stores out * (1 - out) * (target - out) in self.delta.

        Stores the result in self.delta. This is called on output nodes
        in backpropagation."""
        self.compute_activation
        out=self.activation
        self.delta=out*(1-out)*(target-out)


    def compute_hidden_delta(self):
        """Stores the weighted sum of next-layer deltas in self.delta.
        NOTE: Be sure to use each edge's old_weight when calculating delta.

        Stores the result in self.delta. This is called on hidden nodes in
        backpropagation. Assumes that the next deltas for nodes in the next
        layer are already up to date."""
        out = self.activation
        outedges = self.out_edges
        wsum = 0.0
        for edge in outedges:
            wsum += edge.old_weight*(edge.dest.delta)
        self.delta = out*(1-out)*wsum



    def update_weights(self):
        """Updates the weight for each incoming edge.

        Computes the new weight based on self.delta and self.learning_rate,
        then calls change_weight on each in-edge.

        Assumes that either compute_output_delta or compute_hidden_delta was
        just called to set self.delta."""

        inedges=self.in_edges
        for edge in inedges:
            weight=edge.weight+self.learning_rate*self.delta*(edge.source.activation)
            edge.change_weight(weight)


class Edge(object):
    """Represents a weighted edge in a neural network.

    Each edge has a source a destination, and a weight. Edges also remember
    their most recent previous weight for computing hidden node deltas."""
    def __init__(self, source, dest, weight_func):
        """Initialize an edge with a random weight.

        weight_func should be a 0-argument function that returns an initial
        weight. In general, this should be a random function so that networks
        are initialized with random weights."""
        self.source = source
        self.dest = dest
        self.weight = weight_func()
        self.old_weight = 0.0

    def change_weight(self, new_weight):
        """Updates self.weight and self.old_weight.

        self.old_weight is needed for computing hidden node deltas during
        backpropagation."""
        self.old_weight = self.weight
        self.weight = new_weight

    def __repr__(self):
        s = "(" + self.source.name + ", "
        s += self.dest.name + ", " + str(self.weight) + ")"
        return s

class Network(object):
    """Represents a densely connected feed-forward neural network."""
    def __init__(self, input_size, output_size, hidden_sizes=[],
                 learning_rate=2.0, weight_scale=0.1, converge=0.01,
                 random_seed=None):
        """Initializes a dense multi-layer neural network.

        input_size: number of input nodes
        output_size: numer of output nodes
        hidden_sizes: list with the number of hidden nodes in each hidden layer
        learning_rate: passed along to the sigmoid hidden and output nodes
        weight_scale: 'scale' parameter of the numpy.random.normal function
                      to be used when generating initial weights.
        converge: threshold for considering the network's outputs correct.
                  the train function will stop before the maximum number of
                  iterations if all outputs are within converge of their
                  targets.
        random_seed: for reproducability, the seed for the pseudorandom number
                     generator can be specified. If random_seed=None, the
                     numpy seed() function will not be called. If seed is not
                     none, it will be passed to the numpy seed() function
                     before initializing edge weights.

        self.layers will be a list of lists of nodes. Each inner list
        corresponds to one layer of the neural network. The first list is the
        input layer, consisting of InputNode objects. The last list is the
        output layer; in between are hidden layers. All nodes in hidden and
        output layers are SigmoidNode objects.

        Edges are not stored directly by the network, but rather in the edge
        lists for each node. Layers are densely connected, so that all nodes in
        layer i are connected to all nodes in layer i+1.
        """
        if random_seed is not None:
            seed(random_seed)

        weight_func = partial(normal, 0, weight_scale)
        self.converge = converge
        self.layers = []
        self.layers.append([InputNode("in_" + str(i)) for i in range(input_size)])
        for i,layer_size in enumerate(hidden_sizes):
            self.layers.append([SigmoidNode("hidden_"+str(i)+"-"+str(j),
                                learning_rate) for j in range(layer_size)])
        self.layers.append([SigmoidNode("out_"+str(i), learning_rate) for i
                            in range(output_size)])

        # densely connect consecutive layers
        for source_layer, dest_layer in zip(self.layers, self.layers[1:]):
            for source, dest in product(source_layer, dest_layer):
                edge = Edge(source, dest, weight_func)
                source.out_edges.append(edge)
                dest.in_edges.append(edge)

        # connect each node to bias
        self.bias = BiasNode()
        for layer in self.layers[1:]:
            for node in layer:
                e = Edge(self.bias, node, weight_func)
                node.in_edges.append(e)
                self.bias.out_edges.append(e)

    def predict(self, input_vector):
        """Computes the network's output for a given input_vector.

        input_vector: activation value for each input node
        returns: a vector of activation values for each output node

        Sets the activation of each node in the input layer to the appropriate
        value in the input vector. Then for each subsequent layer, calls each
        node's compute_activation function to update its activation. Collects
        the activation values for each output node into the return vector.
        """
        inputlayer = self.layers[0]
        for i in range(len(inputlayer)):
            inputlayer[i].activation = input_vector[i]

        layers = self.layers
        output_vector=[]
        a=0
        #skips input layer

        #i=current layer
        for i in range(1,len(layers)):
            #j is the node we are on in the layer
            for j in range(len(layers[i])):
                layers[i][j].compute_activation()
                if  i==len(layers)-1:
                    output_vector.append(layers[i][j].activation)
        return output_vector

    def backpropagation(self, target_vector):
        """Updates all weights for a single step of stochasticgradient descent.

        Assumes that predict has just been called on the input vector
        corresponding to the given target_vector.

        target_vector: expected activation for each output
        returns: nothing

        Calls compute_output_delta on each node in the output layer, then
        working backwards, calls compute_hidden_delta on all nodes in each
        hidden layer.
        """

        layers=self.layers
        outputlayer=self.layers[len(self.layers)-1]
        for i in range(len(outputlayer)):
            self.layers[len(self.layers)-1][i].compute_output_delta(target_vector[i])
            self.layers[len(self.layers)-1][i].update_weights()


        #j is the layer
        for j in range(len(layers)-2,0,-1):
            for k in range(len(layers[j])):
                self.layers[j][k].compute_hidden_delta()
                self.layers[j][k].update_weights()






    def train(self, data_set, epochs, verbose=False, random_seed=None):
        """Runs repeated prediction & backpropagation steps to learn the data.

        data_set: a list of (input_vector, target_vector) pairs
        epochs: maximum number of times to loop through the data set.
        verbose: if False, nothing is printed; if True, prints the epoch on
                 on which training converged (or that it didn't)
        for reproducability, the seed for the pseudorandom number
                     generator can be specified. If random_seed=None, the
                     numpy seed() function will not be called. If seed is not
                     none, it will be passed to the numpy seed() function
                     before the first time the data set is shuffled.
        returns: nothing

        Runs iterations loops through the data set (shuffled each time).
        Each loop runs predict to compute activations, then backpropagation
        to update weights on every example in the data set. If all outputs
        are within self.converge of their targets (self.test returns 1.0)
        training stops regardless of the iteration.
        """
        if random_seed is not None:
            seed(random_seed)

        for i in range(epochs):
          print(i)
          shuffle(data_set)

          #check if training is done
          """
          if self.test(data_set) == 1.0:
            if verbose:
              print("\nTraining converged on epoch", i)
            return
          """
          #loop through each vector/ layer
          for input, target_output in data_set:

            output = self.predict(input)

            self.backpropagation(target_output)



    def test(self, data_set):
        """Predicts every input in the data set and returns the accuracy.

        data_set: a list of (input_vector, target_vector) pairs
        returns: accuracy, the fraction of output vectors within self.converge
                 of their targets (as measured by numpy.allclose's rtol).

        Calls predict() on each input vector in the data set, and compares the
        result to the corresponding target vector from the data set.
        """

        correct = 0.0
        total = 0.0

        for input, target in data_set:
          #actual output from neural net
          output = self.predict(input)
          total += 1.0 #number of total output vectors

          if allclose(output, target, self.converge) == True:
            correct += 1.0

        return correct/total




    def __repr__(self):
        s = "Neural Network\n"
        for layer in self.layers:
            s += repr(layer) + "\n"
            for node in layer:
                if node.out_edges:
                    s += "  " + repr(node.out_edges) + "\n"
        s += "  " + repr(self.bias.out_edges)
        return s

def main():
    # train a network with one hidden layer on XOR
    data_set = [([0,0],[0.1]), ([0,1],[0.9]), ([1,0],[0.9]), ([1,1],[0.1])]
    nn = Network(2, 1, [2], random_seed=0)
    nn.train(data_set, 10000, verbose=True, random_seed=0)
    print("\nresulting", nn, "\n")
    print("accuracy =", nn.test(data_set))
    for input_vector, target_vector in data_set:
        output_vector = nn.predict(input_vector)
        print("input:", input_vector, "target:", target_vector, "output:", output_vector)

if __name__ == "__main__":
    main()
