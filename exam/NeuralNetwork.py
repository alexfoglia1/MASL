import random
import math

class NeuralNetwork:
    LEARNING_RATE = 0.5
    def __init__(self,n_in,n_hid,n_out,hid_bias,out_bias):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.hidden = NeuronLayer(self.n_hid,hid_bias)
        self.output = NeuronLayer(self.n_out,out_bias)
        self.__in_to_hid__()
        self.__hid_to_out__()
    def __in_to_hid__(self):
        for _hidden in range(self.n_hid):
            for _input in range(self.n_in):
                self.hidden.neurons[_hidden].weights.append(random.random())
    def __hid_to_out__(self):
        for _output in range(self.n_out):
            for _hidden in range(self.n_hid):
                self.output.neurons[_output].weights.append(random.random())
    def inspect(self):
        print("------\nNeural Network:")
        print("Inputs: {}".format(self.n_in))
        print("------\nHidden Layer:")
        self.hidden.inspect()
        print("------\nOutput Layer:")
        self.output.inspect()
        print("------")
    def feedForward(self, inputs):
        hidden_outputs = self.hidden.feedForward(inputs)
        return self.output.feedForward(hidden_outputs)
    def train(self, training_in,training_out):
        self.feedForward(training_in)
         # 1. Output neuron deltas
        deltaWrtOut = [0] * len(self.output.neurons)
        for out in range(len(self.output.neurons)):
            deltaWrtOut[out] = self.output.neurons[out].calculateDeltaWrtTotalIn(training_out[out])
        # 2. Hidden neuron deltas
        deltaWrtHid = [0] * len(self.hidden.neurons)
        for hid in range(len(self.hidden.neurons)):
            deltaWrtHiddenOut = 0
            for out in range(len(self.output.neurons)):
                deltaWrtHiddenOut += deltaWrtOut[out] * self.output.neurons[out].weights[hid]
            deltaWrtHid[hid] = deltaWrtHiddenOut * self.hidden.neurons[hid].calculateDeltaWrtIn()
        # 3. Update output neuron weights
        for out in range(len(self.output.neurons)):
            for w_ho in range(len(self.output.neurons[out].weights)):
                deltaWrtWeight = deltaWrtOut[out] * self.output.neurons[out].calculateDeltaWrtWeight(w_ho)
                self.output.neurons[out].weights[w_ho] -= self.LEARNING_RATE * deltaWrtWeight
        # 4. Update hidden neuron weights
        for hid in range(len(self.hidden.neurons)):
            for w_ih in range(len(self.hidden.neurons[hid].weights)):
                deltaWrtWeight = deltaWrtHid[hid] * self.hidden.neurons[hid].calculateDeltaWrtWeight(w_ih)
                self.hidden.neurons[hid].weights[w_ih] -= self.LEARNING_RATE * deltaWrtWeight
    def totalError(self, training_sets):
        err = 0
        for t in range(len(training_sets)):
            t_in,t_out = training_sets[t]
            self.feedForward(t_in)
            for out in range(len(t_out)):
                err += self.output.neurons[out].calculateError(t_out[out])
        return err

class NeuronLayer:
    def __init__(self, n, bias):
        self.neurons = []
        for i in range(n):
            self.neurons.append(Neuron(bias))
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for neuron in range(len(self.neurons)):
            print(' Neuron', neuron)
            for weight in range(len(self.neurons[neuron].weights)):
                print('  Weight:', self.neurons[neuron].weights[weight])
    def feedForward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.getOut(inputs))
        return outputs
    def getOutputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self,bias):
        self.squash = lambda x : 1/(1+math.exp(-x))
        self.weights = []
        self.bias = bias
    def getOut(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.__comb__lin__inputs__())
        return self.output
    def __comb__lin__inputs__(self):
        res = 0
        for i in range(len(self.inputs)):
            res += self.inputs[i] * self.weights[i]
        return res + self.bias
    def calculateDeltaWrtTotalIn(self, target_output):
        return self.calculateDeltaWrtOut(target_output) * self.calculateDeltaWrtIn();
    def calculateError(self, target_output):
        return 0.5 * (target_output - self.output) ** 2
    def calculateDeltaWrtOut(self, target_output):
        return -(target_output - self.output)
    def calculateDeltaWrtIn(self):
        return self.output * (1 - self.output)
    def calculateDeltaWrtWeight(self, i):
        return self.inputs[i]

#Example Main             
#Define training_sets [[[expected inputs],[expected outputs]],...]
#nn = NeuralNetwork(n_inputs,n_hidden,n_outputs,hid_bias,out_bias)
#for i in range(10000):
#    t_in,t_out = random.choice(training_sets)
#    nn.train(t_in,t_out)
#print("Total error: "+str(nn.totalError(training_sets)))

    
                    
        
