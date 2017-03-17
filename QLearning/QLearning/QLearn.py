import numpy as np
import numpy.random as rand
import os

class QLearn(object):
    def __init__(self, size=10, obstacles=None, biggulps=None):
        """
        Parameters
        ----------
        size      : int : represents length and width of environment space
        obstacles : [((m_start, n_start), (m_end, n_end))] : list of start and end points for walls 
                    to be placed in the environment -- part 5 of the assignment
        """
        self.agent_sensors = ["North", "South", "East", "West", "Here"]
        self.sensor_values = ["Empty", "Can", "Wall"]
        if biggulps is not None: self.sensor_values.append("BigGulp")
        self.size = size
        q_idx_max = len(self.sensor_values)**len(self.agent_sensors)
        self.grid = np.zeros_like(np.arange(size**2, dtype=np.float64).reshape((size, size)))
        self.qmatrix   = np.zeros_like(np.arange(q_idx_max*len(self.agent_sensors), dtype=np.float64).reshape(q_idx_max,len(self.agent_sensors)))
        self.obstacles = obstacles
        self.dinglebs  = biggulps
    
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
        g      : (int, int) : represents a grid position
        action : str        : action for n offset lookup

        Returns
        --------
        q_idx : int : row index to self.qmatrix
        """
        # north-sensor
        if g[0] == 0: north = 2
        else: north = self.grid[g[0]-1][g[1]]

        # east-sensor
        if g[1] == self.size-1: east = 2
        else: east = self.grid[g[0]][g[1]+1]

        # south-sensor
        if g[0] == self.size-1: south = 2
        else: south = self.grid[g[0]+1][g[1]]

        # west-sensor
        if g[1] == 0: west = 2
        else: west = self.grid[g[0]][g[1]-1]

        # here-sensor
        here = self.grid[g[0]][g[1]]

        q_code  = [north, south, east, west, here]
        s_count = len(self.sensor_values)
        q_idx   = [p*s_count**i for i, p in enumerate(q_code, 0)]
        return np.sum(q_idx)
    
    def perform_action(self, g, action):
        """
        Parameters
        ----------
        g : (int, int)    : grid square
        a : string or int : action

        Returns
        -------
        g_prime : (int, int) : new grid square
        r       : int        : reward
        move    : string     : move to make
        """

        # move north
        north = "North"
        if action == north or action == 0:
            if g[0] == 0: return g, -5, north
            else: return [g[0]-1, g[1]], 0, north

        # move east
        east = "East"
        if action == east or action == 1:
           if g[1] == self.size-1: return g, -5, east
           else: return [g[0], g[1]+1], 0, east

        # move south
        south = "South"
        if action == south or action == 2:
            if g[0] == self.size-1: return g, -5, south
            else: return [g[0]+1, g[1]], 0, south

        # move west
        west = "West"
        if action == west or action == 3:
            if g[1] == 0: return g, -5, west
            else: return [g[0], g[1]-1], 0, west

        # pick up can
        pick_up_can = "Here"
        if action == pick_up_can or action == 4:
            if self.grid[g[0]][g[1]]   == 1: return g, 10, pick_up_can
            elif self.grid[g[0]][g[1]] == 3: return g, 20, pick_up_can 
            else: return g, -1, pick_up_can

    def get_move(self, g, random):
        """
        Parameters
        ----------
        g      : tuple : (x, y) of grid square
        random : bool  : true for random action, false for best

        Returns
        --------
        qsa    : float      : the qvalue for the action taken
        gp     : (int, int) : the next grid point
        r      : int        : reward value
        action : string     : action
        """
        if random:
            action = rand.randint(0,len(self.agent_sensors))          
            qsa    = self.qmatrix[self.get_state(g), action]
            gp,r,a = self.perform_action(g, action)
        else:  
            q_values = self.qmatrix[self.get_state(g)]
            q_argmax, qsa = np.argmax(q_values), np.max(q_values)
            gp,r,a = self.perform_action(g, q_argmax)
        
        return qsa, gp, r, a

    def action_dictionary(self, x):
        return {
            "North" : 0,
            "East"  : 1,
            "South" : 2,
            "West"  : 3,
            "Here"  : 4,
        }[x]

    def learn(self, 
              epsilon=1., eps_reduction=.01, eps_const=False, eps_red_interval=50,
              N=5000, M=200, eta=0.2, gamma=0.9, tax=None):
        self.rewards, self.epsilon = [0], epsilon
        for episode in range(N):
            os.system('cls')
            print(" episode: %d\n epsilon: %.3f\n reward:  %d" % \
                (episode, self.epsilon, self.rewards[-1]))

            self.randomize_grid()
            reward = 0
            g = [rand.randint(0, 10), rand.randint(0, 10)]

            for step in range(1, M):   
                # choose an action based on eps-greedy action selection
                g_qdx = self.get_state(g)

                # use epsilon greedy
                random_action = rand.choice([True,False], p=[self.epsilon, 1-self.epsilon])
                qsa, gp, r, a = self.get_move(g, random_action)                

                # remove a can if Robby found one
                if r > 0: self.grid[gp[0]][gp[1]] = 0

                # tax Robby for going too slow
                if tax is not None: r -= tax

                # update the qmatrix
                qsp, x, x, x = self.get_move(gp, random=False)
                self.qmatrix[g_qdx][self.action_dictionary(a)] = \
                    qsa + eta * (r + gamma*qsp - qsa)

                reward += r   # accumulate reward   
                g = gp        # update position     
            if not eps_const and self.epsilon > .1 and episode%eps_red_interval == 0: 
                self.epsilon -= eps_reduction
            if episode%1000 == 0: self.rewards.append(reward)