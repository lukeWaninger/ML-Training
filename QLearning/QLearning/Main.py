import QLearn, sys, pandas as pd

def main():
    ## Robby run part 1a... Robby learns
    #robby = QLearn.QLearn()
    #robby.learn()
    #pd.DataFrame(robby.rewards[1:]).to_csv("exp1_train.csv", header=None)    
    
    ## Robby run part 1b... Robby tries so hard
    #robby.learn(epsilon = .1)
    #pd.DataFrame(robby.rewards[1:]).to_csv("exp1_test.csv", header=None)
    
    # Robby run part 2...
    #eta_range = [.2*i for i in range(1, 5)]
    #r_train, r_test = [], []
    #for eta in eta_range:
    #    robby = QLearn.QLearn()
    #    robby.learn(eta = eta)
    #    r_train.append(robby.rewards[1:])

    #    robby.learn(epsilon = .1)
    #    r_test.append(robby.rewards[1:])
    #pd.DataFrame(r_train).to_csv("exp2_test.csv" , header=None)
    #pd.DataFrame(r_test).to_csv("exp2_train.csv", header=None)

    # Robby run part 3...
    #robby = QLearn.QLearn()
    #robby.learn(eps_const=True, epsilon=.9)
    #pd.DataFrame(robby.rewards[1:]).to_csv("exp3_train.csv", header=None)    

    #robby.learn(epsilon = .1)
    #pd.DataFrame(robby.rewards[1:]).to_csv("exp3_test.csv", header=None)

    # Robby run part 4...
    #robby = QLearn.QLearn()
    #robby.learn(tax = .5)
    #pd.DataFrame(robby.rewards[1:]).to_csv("exp4_train.csv", header=None)    
    
    #robby.learn(epsilon = .1)
    #pd.DataFrame(robby.rewards[1:]).to_csv("exp4_test.csv", header=None)

    # Robby run part 5...
    robby = QLearn.QLearn()
    gamma_range = [.2*i for i in range(1,5)]
    r_train, r_test = [], []
    for gamma in gamma_range:
        robby = QLearn.QLearn()
        robby.learn(gamma = gamma)
        r_train.append(robby.rewards[1:])

        robby.learn(epsilon = .1)
        r_test.append(robby.rewards[1:])
    pd.DataFrame(r_train).to_csv("exp5_test.csv" , header=None)
    pd.DataFrame(r_test).to_csv("exp5_train.csv", header=None)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
