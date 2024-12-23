import numpy as np
import matplotlib.pyplot as pyplot

"""This refers to the agent + environment class
This is a grid problem of reinforcement learning.

Initialized:
0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 

state spaces:
1 2  3  4  5  6  7  8
9 10 11 12 13 14 15 16
17 18 19 20 21 22 23 24
25 26 27 28 29 30 31 32


"""
class GridWorld(object):
    def __init__(self, n, m, magicSquares):
        self.grid = np.zeros((m, n))
        self.m = m #breadth of the grid 
        self.n = n #length of the grid
        self.stateSpace = [i for i in range(self.m*self.n)] #state spaces
        self.stateSpace.remove(80)
        self.stateSpacePlus = [i for i in range(self.m*self.m)] #statespaceplus is for the next states
        self.possibleAction = ['U', 'D', 'L', 'R'] #possible actions in the grid are left, right, up, down
        self.actionSpace = {'U':-self.m,
                            'D':self.m,
                            'L':-1,
                            'R': 1} #what the actions lead to 
        self.P = {} #probability space. stores the transition probability
        self.magicSquares = magicSquares
        self.initP()

    #initP method initializes the states and actions and initial transition probabilities
    def initP(self):
        for state in self.stateSpace:
            for action in self.possibleAction:
                reward = -1 #reward is -1 for all the states
                state_ = state + self.actionSpace[action]
                #if the new state is a magic state
                if state_ in self.magicSquares.keys():
                    state_ = self.magicSquares[state_]
                #check is the new state is offgrid
                if self.offGridMove(state_, state):
                    state_ = state
                #check if the new state is a terminal 
                if self.isTerminalState(state_):
                    reward = 0
                #probability of transition, this does not reflect the chances of going from
                #state to state_, this signifies the probability of transition. Which means
                # if there is a situation of going from state to state_, the probability is 1
                # without considering the policy.
                self.P[(state_, reward, state, action)] = 1
    
    def isTerminalState(self, state):
        #if a state is terminal, it will only exist in the state space, and not in future state space(which is stateSpacePlus)
        return state in self.stateSpacePlus and state not in self.stateSpace
    
    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpacePlus:
            return True
        #if we're trying to wrap around to next row
        elif oldState % self.m ==0 and newState % self.m == self.m -1:
            return True
        elif oldState % self.m == self.m -1 and newState % self.m ==0:
            return True
        else:
            return False
        
def printV(V, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            print('%.2f'  % V[state], end='\t')
        print('\n')
    print('-------------------------------')

def printPolicy(policy, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            if not grid.isTerminalState(state):
                if state not in grid.magicSquares.keys():
                    print('%s' % policy[state], end='\t')
                else:
                    print('%s' % '--', end='\t')
            else:
                print('%s' % '--', end='\t')
        print('\n')
    print('--------------------------')

def evaluatePolicy(grid, V, policy, GAMMA, THETA):
    #policy evaluation for the random choice in gridworld
    converged = False
    i = 0
    while not converged:
        DELTA = 0
        for state in grid.stateSpace:
            i +=1
            oldV = V[state]
            total = 0
            weight = 1 / len(policy[state])
            for action in policy[state]:
                for key in grid.P:
                    (newState, reward, oldState, act) = key
                    #we're given state and action, want new state and reward
                    if oldState == state and act == action:
                        total += weight * grid.P[key] * (reward + GAMMA*V[newState])
                
            V[state] = total
            DELTA = max(DELTA, np.abs(oldV - V[state]))
            converged = True if DELTA < THETA else False
    print(i, 'sweeps of state space in policy evaluation')
    return V
    
def improvePolicy(grid, V, policy, GAMMA):
    stable = False
    newPolicy = {}
    i = 0
    for state in grid.stateSpace:
        i += 1
        oldActions = policy[state]
        value = []
        newAction = []
        for action in policy[state]: #for each action in a state according to a policy
            weight = 1/len(policy[state]) #number of possible actions\
            for key in grid.P: #for each tuple of (newstate, reward, oldstate, action) in transition probability space
                (newstate, reward, oldstate, act) = key #open all the variables

                # these variables contain all the possible states and rewards, so we will right now
                # only use the probabilites for the current state and currenta action
                if oldstate == state and act == action:
                    value.append(np.round(weight*grid.P[key]*(reward + GAMMA*V[newstate]), 2))
                    newAction.append(action)
        
        value = np.array(value)
        best = np.where(value == value.max())[0]
        bestActions = [newAction[item] for item in best] #best actions are where value is maximized

        newPolicy[state] = bestActions

        if oldActions != bestActions:
            stable = False
        
    print(i, 'sweeps of state spaces in policy improvement')
    return 

if __name__ == '__main__':
    magicSquares = {18: 54, 63: 14}
    env = GridWorld(9, 9, magicSquares)

    #moodel hyperparameters
    GAMMA = 1.0
    THETA = 1e-6

    V = {}
    for state in env.stateSpacePlus:
        V[state] = 0
    
    policy = {}
    for state in env.stateSpace:
        policy[state] =  env.possibleAction

    stable =False
    while not stable:

        V = evaluatePolicy(env, V, policy, GAMMA, THETA)  
        stable, policy = improvePolicy(env, V, policy, GAMMA)
    print(V, env)
    printPolicy(V, env)

