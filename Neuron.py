#! /usr/bin/python3
import random
import numpy as np
from math import e
class One_Input_Neuron:
    def __init__(self,inp):
       self.inputs=np.array([-1,inp])
       self.weights=np.array([random.random(),random.random()])
    def run(self,inp=0,t=False):
        inp1=self.inputs
        weights=self.weights
        if t:
            inp1=np.array([-1,inp])
        i=sum(inp1*weights)
        output=1/(1+ e**(-i))
        if not t:
            self.output=output
        return (output)
    def train(self,x,d,rate):
        output=self.run(x,t=True)
        w1=output*(1-output)*(d-output);
        w2=w1*x;
        self.weights=rate*np.array([w1,w2])

    def Train(self,rate,inputs,outputs):
        if len(inputs)!=len(outputs):
            return 1;
        for i in range(len(inputs)):
            self.train(inputs[i],outputs[i],rate)
        return 0
    #def connect(self,other):
     #   other.__init__(self.output)
if __name__=='__main__':
    neuron1=One_Input_Neuron(2221)
    #neuron2=One_Input_Neuron(2222)
    inp=[]
    out=[]
    for i in range(1000):
        inp.append(i/1000)
        out.append(i/1000)
    test=neuron1.Train(5,inp,out)
    if test!=1:
        print(neuron1.run())
        print(neuron1.weights)
