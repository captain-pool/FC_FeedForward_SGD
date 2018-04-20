#! /usr/bin/python3

'''
Program to Differentiate between Even and Odd numbers of 30 bit 
'''
from FC_SGD import Network
import numpy as np
from _pickle import load
from _pickle import dump
def bits(i):
    assert len(bin(i))<=32
    l=list(bin(i).split('b')[1])
    return np.array(list(map(int,([0]*(30-len(l)))+l)))
def one_hot_vector(i):
    v=np.zeros(2)
    v[i]=1
    return v

def main():
    n=Network([30,40,2])
    l=[]
    #l=np.array([([1,1],[0]),([0,0],[0]),([1,0],[1]),([0,1],[1])])
    for i in range(1,2**9):
        l.append((bits(i),one_hot_vector(i%2)))
    print("Training....")
    n.SGD(np.array(l),1,1,1)
    dump(n,open("EvOddModel.sav","wb"))
    #print('Testing .... \n')
    #for i in range(821,1024):
    #    print(n.feedforward(bits(i)),'---->',one_hot_vector(i%2))
#    print(n.feedforward(bits(3000)),'---->',one_hot_vector(3000%2))
if __name__=='__main__':
    main()
