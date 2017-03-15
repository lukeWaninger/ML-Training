import numpy as np
import numpy.random as rand
import os

class QLearn(object):
    def __init__(self, size=10, obstacles=None, dinglebs=None):
        """
        Parameters
        ----------
        size      : int : represents length and width of environment space
        obstacles : [((m_start, n_start), (m_end, n_end))] : list of start and end points for walls 
                    to be placed in the environment -- part 5 of the assignment
        """
        self.agent_sensors = ["North", "South", "East", "West", "Here"]
        self.sensor_values = ["Empty", "Can", "Wall"]
        self.size = size
        q_idx_max = len(self.sensor_values)**len(self.agent_sensors)
        self.grid = np.zeros_like(np.arange(size**2, dtype=np.float64).reshape((size, size)))
        self.qmatrix   = np.zeros_like(np.arange(q_idx_max*len(self.agent_sensors), dtype=np.float64).reshape(q_idx_max,len(self.agent_sensors)))
        self.obstacles = obstacles
        self.dinglebs  = dinglebs
    
    def randomize_grid(self):
        self.grid = np.zeros_like(self.grid, dtype=np.float64) # reset to zeros
        self.grid = np.random.choice([0.,1.], size=(self.size,self.size), replace=True)

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

    def get_state(self, g):
        """
        Parameters
        ----------
        g : (x,y) : represents a grid position

        Returns
        --------
        q_idx : int : index to self.qmatrix
        """
        # north
        if g[0] == 0: north = 2
        else: north = self.grid[g[0]-1,g[1]]

        # east
        if g[1] == self.size-1: east = 2
        else: east = self.grid[g[0],g[1]+1]

        # south
        if g[0] == self.size-1: south = 2
        else: south = self.grid[g[0]+1,g[1]]

        # west
        if g[1] == 0: west = 2
        else: west = self.grid[g[0],g[1]-1]

        # here
        here = self.grid[g]

        return np.sum(np.array([north, east, south, west, here])**len(self.agent_sensors))
    
    def perform_action(self, g, action):
        """
        Parameters
        ----------
        g : (int, int) : grid square
        a : str        : action

        Returns
        -------
        g_prime : (int, int) : new grid square
        r       : int        : reward
        """

        # north
        if action == "North":
            if g[0] == 0: return g, -5
            else: return (g[0]-1, g[1]), 0

        # east
        if action == "East":
           if g[1] == self.size-1: return g, -5
           else: return self.grid[g[0],g[1]+1], 0

        # south
        if action == "South":
            if g[0] == self.size-1: return g, -5
            else: return self.grid[g[0]+1,g[1]], 0

        # west
        if action == "West":
            if g[1] == 0: return g, -5
            else: return self.grid[g[0],g[1]-1], 0

        # pick up can
        if action == "Here":
            if self.grid[g] == 1: return g, 10
            else: return g, -1

    def get_move(self, g, random=False):
        """
        Parameters
        ----------
        g : tuple : (x, y) of grid square

        Returns
        --------
        qsa, (x,y)', r
        """
        if random:
            action = self.agent_sensors[rand.randint(0,5)]
            g, r   = self.perform_action(g, action)
        else:
            actions  = np.unique(np.array([perform_action(a) for a in self.sensor_values])[:,:0])
            q_values = 

        # north - east - south - west - here
        actions= []
        states = [self.get_state(m) for m in moves]
        values = [self.qmatrix[s] for s in states]

        if r > 0: self.grid[g] = 0
        qsa = self.qmatrix[self.get_state(g)]
        return values[arg], locations[arg]

    def learn(self, 
              epsilon=1., eps_reduction=.01, eps_const=False, eps_red_interval=50,
              N=5000, M=200, eta=0.2, gamma=0.9, tax=None):
        self.rewards, self.epsilon = [0], epsilon
        for episode in range(N):
            os.system('cls')
            print(" episode: %.3f\n epsilon: %.3f\n reward:  %.3f" % \
                (episode, self.epsilon, self.rewards[-1]))

            self.randomize_grid()
            reward = 0.

            # Robby is magically conceived in a glowing ball of energy with a robotic Austrian accent
            g = (rand.randint(0, 10), rand.randint(0, 10))

            # Robby busts a move
            for step in range(1, M):   
                # choose an action based on eps-greedy action selection
                cur_state = self.get_state(g)

                # use epsilon greedy
                random_action = rand.choice([True,False], p=[self.epsilon, 1-self.epsilon])
                qsa, gp = self.get_move(g, random_action)                

                r = self.grid[gp]
                # remove the can from Robby's environment
                if r > 0: self.grid[gp] = 0.

                # tax Robby for going too slow
                if tax is not None: r -= tax

                # update the qmatrix
                qsp, none = self.get_move(s)
                self.qmatrix[cur_state] = qsa + eta * (r + gamma*qsp - qsa)

                reward += r   # accumulate reward   
                g = gp             
            if not eps_const and self.epsilon > .1 and episode%eps_red_interval == 0: 
                self.epsilon -= eps_reduction
            if episode%100 == 0: self.rewards.append(reward)