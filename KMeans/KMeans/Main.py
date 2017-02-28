import pandas as pd
import numpy  as np
import sys

def main():
    # read in the data
    X_train = pd.read_csv("optdigits.train").values
    X_test  = pd.read_csv("optdigits.test").values
    print()
     
if __name__ == "__main__":
    sys.exit(int(main() or 0))