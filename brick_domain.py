import numpy as np
import numpy.random as rnd
from numpy.linalg import norm
from rlpy.Domains.Domain import Domain

import matplotlib.pyplot as plt

class StateIndex:
    X = 0
    XDOT = 1

def to_range(x, rng):
    '''Map a variable in range [0,1] to the given range pair'''
    assert rng[1] > rng[0]      # Could also use >=, but that's a sorta odd use case
    return x*(rng[1] - rng[0]) + rng[0]

def in_range(x, rng):
    '''Return true if x lies in the closed set given by range, otherwise return false'''
    assert rng[1] > rng[0]
    return x <= rng[1] and x >= rng[0]    

class BrickDomain(Domain):

    pos_limits = np.array((-10.0, 10.0))
    vel_limits = np.array((-30.0, 30.0))
    actions = np.array((-100.0, 100.0))

    episode_length = 100        # Number of steps, not a physical time
    dt = 0.01
    
    discount_factor = 0.9

    random_start = True
    default_state = np.array((0.0, 0.0))

    goal_pos = 0.0
    goal_tolerance = 0.5

    visualization_bins = 20
    visualization_labels = 5

    def __init__(self):

        # Things that need to be defined for the rlpy domain class
        self.statespace_limits = np.array((self.pos_limits, self.vel_limits))
        self.continuous_dims = [StateIndex.X, StateIndex.XDOT]
        self.episodeCap = self.episode_length
        self.actions_num = len(self.actions)
        
        super(BrickDomain, self).__init__()

    # Abstract method for Domain class
    def s0(self):
        if self.random_start:
            self.state = self._randomState()
        else:
            self.state = self.default_state

        self.traj = [self.state]

        self.current_step = 0
        # Implicitly converted to tuple
        # print "Step ", self.current_step, " is terminal: ", self.isTerminal(), " state ", self.state
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    # Abstract method for Domain class
    def step(self, action_id):
        
        action = self.actions[action_id]
        # print action
        
        state = self.state.copy()
        state[0] += state[1]*self.dt
        state[1] += action*self.dt
        self.current_step += 1
        # print self.current_step
        # print state

        new_state = state
        # print new_state
        self.state = new_state
        self.traj.append(new_state)
        reward = self._getReward(action)
        
        return (reward, new_state, self.isTerminal(), self.possibleActions() )

    def isTerminal(self, s = None):
        if s is None:
            s = self.state
        '''True if state lies within limits, otherwise false'''
        return not (in_range(s[0], self.pos_limits) and in_range(s[1], self.vel_limits) )

    def _randomState(self):
        return np.array([to_range(self.random_state.rand(), self.pos_limits), 
                         to_range(self.random_state.rand(), self.vel_limits)])
    
    def _getReward(self, action):
        if self.isTerminal():
            return -1.0*(self.episode_length - self.current_step)
        elif self._atGoal():
            return 1.0
        else:
            return -0.01*self.state[0]**2

    def _atGoal(self):
        return abs(self.state[0] - self.goal_pos) < self.goal_tolerance

    def showDomain(self, a):
        traj = np.array(self.traj)

        plt.figure(1)
        plt.clf()
        
        plt.scatter(traj[:,0], traj[:,1], c=range(len(traj)))
        plt.xlim(self.pos_limits[0], self.pos_limits[1])
        plt.ylim(self.vel_limits[0], self.vel_limits[1])
        plt.draw()
        
    def showLearning(self, representation):
        
        x = np.linspace(self.pos_limits[0], self.pos_limits[1], self.visualization_bins)
        xdot = np.linspace(self.vel_limits[0], self.vel_limits[1], self.visualization_bins)

        xticks = np.arange(self.visualization_labels)*float(self.visualization_bins)/(self.visualization_labels-1)
        yticks = xticks

        xlabels = ['{0:.1f}'.format(xx) for xx in np.linspace(self.pos_limits[0], self.pos_limits[1], self.visualization_labels)]
        ylabels = ['{0:.1f}'.format(yy) for yy in np.linspace(self.vel_limits[0], self.vel_limits[1], self.visualization_labels)]

        V = np.zeros((len(xdot), len(x)))
        pi = np.zeros((len(xdot), len(x)))

        for (idx, xx) in enumerate(x):
            for (idy, yy) in enumerate(xdot):
                s = np.array((xx, yy))
                term = self.isTerminal(s)
                Qs = representation.Qs(s, term) # Value for all possible actions in state s
                As = self.possibleActions()
                # If multiple optimal actions, pick one randomly
                a = np.random.choice(As[Qs.max() == Qs])
                pi[idy, idx] = a
                V[idy, idx] = max(Qs)

        plt.figure(2)
        plt.clf()
        plt.imshow(V, origin='lower')
        plt.title('Value Function')
        plt.colorbar()
        plt.xticks(xticks, xlabels)
        plt.yticks(yticks, ylabels)
        plt.figure(3)
        plt.clf()
        plt.imshow(pi, origin='lower')
        plt.title('Policy')
        plt.xticks(xticks, xlabels)
        plt.yticks(yticks, ylabels)
        plt.colorbar()
        plt.draw()

