import QLearn, sys, pandas as pd

def main():
    part_one()
    #part_two()
    #part_three()
    #part_four()
    #part_five()

def part_one():
    # train Robby train. part one
    robby = QLearn.QLearn()
    robby.learn()
    pd.DataFrame(robby.rewards[1:]).to_csv("exp1_train.csv", header=None)    

    # Robby runs
    robby.learn(epsilon = .1)
    pd.DataFrame(robby.rewards[1:]).to_csv("exp1_test.csv", header=None)

def part_two():
    eta_range = [.2*i for i in range(1, 5)]
    r_train, r_test = [], []
    for eta in eta_range:
        #Robby learns
        robby = QLearn.QLearn()
        robby.learn(eta = eta)
        r_train.append(robby.rewards[1:])

        # Robby runs
        robby.learn(epsilon = .1)
        r_test.append(robby.rewards[1:])

    # write Robby's results
    pd.DataFrame(r_train).to_csv("exp2_test.csv" , header=None)
    pd.DataFrame(r_test).to_csv("exp2_train.csv", header=None)

def part_three():
    # Robby run part 3...
    robby = QLearn.QLearn()
    robby.learn(eps_const=True, epsilon=.9)
    pd.DataFrame(robby.rewards[1:]).to_csv("exp3_train.csv", header=None)    

    # testing
    robby.learn(epsilon = .1)
    pd.DataFrame(robby.rewards[1:]).to_csv("exp3_test.csv", header=None)

def part_four():
    # Robby run part 4...
    robby = QLearn.QLearn()
    robby.learn(tax = .5)
    pd.DataFrame(robby.rewards[1:]).to_csv("exp4_train.csv", header=None)    

    # test
    robby.learn(epsilon = .1)
    pd.DataFrame(robby.rewards[1:]).to_csv("exp4_test.csv", header=None)

def part_five():
    # Robby run part 5...
    walls    = [((3,4),(3,9)),
             ((5,4),(5,7)),
             ((5,7),(8,7)),
             ((3,9),(8,9)),
             ((9,2),(13,2)),
             ((9,4),(13,4)),
             ((3,16),(8,16)),
             ((3,18),(8,18)),
             ((16,9),(16,15)),
             ((18,9),(18,15))]
    dinglebs = [(4,7),(4,8),(5,8),(17,11),(17,12),(17,13)]
    robby    = QLearn.QLearn(size=20,obstacles=walls,dinglebs=dinglebs)
    
    N, N_inc, M, eps_reduction, eps_red_interval = 5000, 1000, 400, .001, 84
    max_n = int(N/N_inc)
    prev_eps = 1
    for i in range(max_n):
        robby.learn(epsilon=prev_eps, 
                    eps_reduction=eps_reduction, 
                    eps_red_interval=eps_red_interval,
                    N=N_inc, M=M)
        with open('exp5_train__N%s_M%s_EReduce%s.csv' % (N, M, eps_reduction), 'a') as f:
            pd.DataFrame(robby.rewards[1:]).to_csv(f, header=None)
            pd.DataFrame(robby.qmatrix).to_csv("exp5_qmatrix__N%s_M%s_EReduce%s.csv" % \
                (N, M, eps_reduction), header=None)
        prev_eps = robby.epsilon
    robby.learn(epsilon=.1)
    pd.DataFrame(robby.rewards[1:]).to_csv("exp5_test__N%s_M%s_EReduce%s.csv" % \
       (N, M, eps_reduction), header=None)

if __name__ == "__main__":
    sys.exit(int(main() or 0))