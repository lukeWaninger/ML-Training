import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loc = "C:\\Users\\Luke\\OneDrive\\School\\CS 445 [Machine Learning]\\Homework\\Homework 6 - QLearning\\content\\"
data = pd.read_csv(loc + "exp5_qmatrix_N50000_M400_EReduce0.001_tax0.5.csv", header=None).values
qs = [data[421][1:], data[161][1:], data[330][1:], data[64][1:], data[0][1:]]
plt.imshow(qs)
plt.savefig(loc + "qs")