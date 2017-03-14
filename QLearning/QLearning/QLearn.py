import numpy as np
import numpy.random as rand
import os

class QLearn(object):
    def __init__(self, size=(10,10), obstacles=None, dinglebs=None):
        """
        Parameters
        ----------
        size      : 2-tuple (int, int) : represents length and width of environment space
        obstacles : [((m_start, n_start), (m_end, n_end))] : list of start and end points for walls 
                    to be placed in the environment -- part 5 of the assignment
        """
        self.agent_actions = ["Move-North", "Move-South", "Move-East", "Move-West", "Pick-Up-Can"]
        self.size = size
        self.grid = np.zeros_like(np.arange(size[0]*size[1]).reshape(size))
        self.qmatrix   = np.zeros_like(np.arange(size[0]*size[1]*len(self.agent_actions)).reshape(size[0]*size[1],len(self.agent_actions)))
        self.obstacles = obstacles
        self.dinglebs  = dinglebs
    
    def randomize_grid(self):
        self.grid = np.zeros_like(self.grid) # reset to zeros
        self.grid = np.random.choice([0,1], size=self.size, replace=True)

        # build the inner walls
        if self.obstacles is not None:
            for wall in self.obstacles:
                start, end = wall[0], wall[1]
                # build a horizontal wall
                if start[0] == end[0]:
                    for n in range(start[1], end[1]+1):
                        self.grid[start[0], n] = 2
                # build a vertical wall
                elif start[1] == end[1]:
                    for m in range(start[0], end[0]+1):
                        self.grid[m, start[1]] = 2 
        
        # place the dingle-berries
        if self.dinglebs is not None:
            for db in self.dinglebs:
                self.grid[db] = 3

    def act(self, s, a):
        """
        Parameters
        ----------
        state  : tuple : represents Robby's position in the grid (m,n)
        action : int   : bounds [0, len(self.actions))

        Return Values
        --------------
        a, s', r : str, (int, int), float : action, next state, reward
        """
               
        # check if Robby hits a boundary wall
        b = self.size[1]-1
        if (s[0] == 0 and a == "Move-North") or \
           (s[0] == b and a == "Move-South") or \
           (s[1] == 0 and a == "Move-West")  or \
           (s[1] == b and a == "Move-East"):    
                return a, s, -5

        # check if Robby hits an inner wall
        if (a == "Move-North" and self.grid[s[0]-1, s[1]] == 2) or \
           (a == "Move-South" and self.grid[s[0]+1, s[1]] == 2) or \
           (a == "Move-West"  and self.grid[s[0], s[1]-1] == 2) or \
           (a == "Move-East"  and self.grid[s[0], s[1]+1] == 2):
                return a, s, -5

        # check if Robby attempts to pick up a can which isn't there
        if a == "Pick-Up-Can" and \
          self.grid[s] != 1   and \
          self.grid[s] != 3: 
            return a, s, -1

        # check if Robby picks up a can
        if a == "Pick-Up-Can" and self.grid[s] == 1: 
            self.grid[s] = 0 # remove the can from Robby's environment
            return a, s, 10
        
        # check if Robby picks up a dingle-berry
        if a == "Pick-Up-Can" and self.grid[s] == 3:
            self.grid[s] = 0
            return a, s, 20 

        # if control flow makes it this far, Robby just made a legal transition
        if a == "Move-North": return a, (s[0]-1, s[1]), 0
        if a == "Move-South": return a, (s[0]+1, s[1]), 0
        if a == "Move-West" : return a, (s[0], s[1]-1), 0
        if a == "Move-East" : return a, (s[0], s[1]+1), 0

    def state_to_qint(self, state):
        return int(str(state[0]) + str(state[1]))

    def action_to_matrix(self, action):
        return (self.state_to_qint(action[1]), self.action_int(action[0]))

    def action_int(self, a):
        return {
            "Move-North" : 0,
            "Move-South" : 1,
            "Move-East"  : 2,
            "Move-West"  : 3,
            "Pick-Up-Can": 4,
        }[a]   

    def qsp(self, s):
        actions = [self.act(s, a) for a in self.agent_actions]
        action  = actions[np.argmax([a[2] for a in actions])]
        return self.qmatrix[self.action_to_matrix(action)]

    def learn(self, epsilon=1., N=5000, M=200, eta=0.2, gamma=0.9, eps_const=False, tax=None):
        self.rewards, self.epsilon = [0], epsilon
        for episode in range(N):
            os.system('cls')
            print(" episode: %d\n epsilon: %.2f\n reward:  %d" % \
                (episode, self.epsilon, self.rewards[-1]))

            self.randomize_grid()
            reward = 0

            # Robby is magically conceived in a glowing ball of energy with a robotic Austrian accent
            s = (rand.randint(0, 10), rand.randint(0, 10))

            # Robby busts a move
            for step in range(1, M):   
                # choose an action based on eps-greedy action selection
                actions = [self.act(s, a) for a in self.agent_actions]

                # use epsilon greedy
                random_action = rand.choice([1,0], p=[self.epsilon, 1-self.epsilon])
                if random_action: 
                    action = actions[rand.randint(0, len(actions))]
                else: 
                    action = actions[np.argmax([a[2] for a in actions])]

                # update the q-matrix
                idx = self.action_to_matrix(action)
                qsa, qsp = self.qmatrix[idx], self.qsp(action[1])
                r   = action[2]
                if tax is not None: r -= tax
                self.qmatrix[idx] = qsa + eta * (r + gamma*qsp - qsa)
                
                s = action[1] # set the next state
                reward += r   # accumulate reward                
            if not eps_const and self.epsilon > .1 and episode%50 == 0: self.epsilon -= .01
            if episode%100 == 0: self.rewards.append(reward)