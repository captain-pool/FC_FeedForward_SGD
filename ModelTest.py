#! /usr/bin/python3
from _pickle import load
from Test import bits,one_hot_vector
import sys
if len(sys.argv)>1:
    n=int(sys.argv[1])
    model=load(open("EvOddModel.sav","rb"))
    output=model.feedforward(bits(n),True)
    print(output)
    print("Probability of being Even:",round(output[0]*100,2),"% \nProbability of being Odd:",round(output[1]*100,2),"%")
else:
    print(sys.argv[0],"<Number>");
