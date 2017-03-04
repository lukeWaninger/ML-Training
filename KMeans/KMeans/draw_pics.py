import matplotlib.pyplot as plt
import pandas            as pd
import os, sys, itertools

data = pd.read_hdf("spambase.hdf", header = None)