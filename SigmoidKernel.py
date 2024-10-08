# 作者:     wxf

# 开发时间: 2022/1/11 9:30
from numpy import *
import numpy as np
import math     # math.pow(x, y)

def SigmoidKernelDisease(DiseaseAndRNABinary):
    DiseasePolynomialKernel = []
    counter = 0
    while counter < len(DiseaseAndRNABinary):
        row = []
        counter1 = 0
        while counter1 < len(DiseaseAndRNABinary):
            sum = 0
            counter2 = 0
            while counter2 < len(DiseaseAndRNABinary[counter]):
                sum = sum + DiseaseAndRNABinary[counter][counter2] * DiseaseAndRNABinary[counter1][counter2]
                counter2 = counter2 + 1
            sum = sum / (753/len(DiseaseAndRNABinary))
            sum = (math.exp(sum) - math.exp(-sum)) / (math.exp(sum) + math.exp(-sum))
            row.append(sum)
            counter1 = counter1 + 1
        counter = counter + 1
        print(counter)
    return DiseasePolynomialKernel

def SigmoidKernelRNA(DiseaseAndRNABinary):
    RNAAndDiseaseBinary = DiseaseAndRNABinary.T
    counter = 0
    while counter < len(RNAAndDiseaseBinary):
        row = []
        counter1 = 0
        while counter1 < len(RNAAndDiseaseBinary):
            sum = 0
            counter2 = 0
            while counter2 < len(RNAAndDiseaseBinary[counter]):
                sum = sum + RNAAndDiseaseBinary[counter][counter2] * RNAAndDiseaseBinary[counter1][counter2]
                counter2 = counter2 + 1
            sum = sum / (753/ len(RNAAndDiseaseBinary))
            sum = (math.exp(sum) - math.exp(-sum)) / (math.exp(sum) + math.exp(-sum))
            row.append(sum)
            counter1 = counter1 + 1
        RNAPolynomialKernel.append(row)
        counter = counter + 1
        print(counter)
    return RNAPolynomialKernel