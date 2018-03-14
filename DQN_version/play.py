import numpy as np
import pylab

from DQN_version.TicTacToe import TicTacToeSingle
from DQN_version.agent import Agent_DQN

if __name__ == "__main__":
    # init TTT Environment
    env = TicTacToeSingle()
    state_size = env.state_size
    action_size = env.action_size
    # init Agent
    agent = Agent_DQN(state_size=state_size, action_size=action_size,load_model=True)
    done = False
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    IsUser = False
    while not done:
        if IsUser:
            action = input("0~~8:")
            action = int(action)
            print('User turn. Action :',action)
        else:
            action = agent.get_action(state)
            print('AI turn. Action :',action)

        next_state, _, done = env.__step__(action)
        env.__printBoard__()
        env.__changeBoardTurn__(env.Board)
        IsUser = not IsUser