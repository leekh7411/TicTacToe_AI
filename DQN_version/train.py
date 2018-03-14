import numpy as np
import pylab
from copy import deepcopy

from DQN_version.TicTacToe import TicTacToeSingle
from DQN_version.agent import Agent_DQN
A = 1
B = -1
EPISODE_NUM = 100
UPDATE = 10
def get_reward(_agentA,_agentB,_Turn,_Env):
    rewardA = 0
    rewardB = 0
    draw = 0
    sim_epsd = 1000
    for e in range(sim_epsd):
        _env = deepcopy(_Env)
        _state = _env.reset()
        _state = np.reshape(_state, [1, _env.state_size])
        _turn = deepcopy(_Turn)
        _winner = 0
        while not _env.GameOver:
            _action = 0
            if _turn == A : _action = _agentA.get_action(_state)
            if _turn == B : _action = _agentB.get_action(_state)
            _next_state , _ , _ = _env.__step__(_action)
            _win_state = _env.__checkWin__(_next_state)
            if _win_state == 1 :
                _winner = _turn
            if _win_state == -1:
                _winner = _turn * -1

            _turn *= -1

        if _winner == A : rewardA += 1
        elif _winner == B : rewardB += 1
        else : draw += 1

    print("Reward (A/B/Draw) :",rewardA,rewardB,draw)
    return rewardA, rewardB

if __name__ == "__main__":
    # init TTT Environment
    env = TicTacToeSingle()
    state_size = env.state_size
    action_size = env.action_size
    # init Agent
    agentA = Agent_DQN(state_size=state_size,action_size=action_size)
    agentB = Agent_DQN(state_size=state_size,action_size=action_size)

    scoresA = []
    scoresB = []
    eA = 0
    eB = 0
    episodesA = []
    episodesB = []

    for e in range(EPISODE_NUM):
        done = False
        scoreA = 0
        scoreB = 0
        state = env.reset()
        state = np.reshape(state,[1,state_size])

        '''
        if e != 0 and e % UPDATE == 0:
            rA1, rB1 = get_reward(agentA,agentB,A,env)
            rA2, rB2 = get_reward(agentA,agentB,B,env)
            if rA1+rA2 > rB1+rB2 :
                agentB.update_model_from(agentA.model)
            else :
                agentA.update_model_from(agentB.model)
        '''
        Turn = 1
        while not done:

            if Turn == A :action = agentA.get_action(state)
            if Turn == B :action = agentB.get_action(state)

            # step
            if env.__actionAvail__(env.Board, action):
                next_state, reward, done = env.__step__(action)
                next_state = np.reshape(next_state,[1,state_size])

                # Reward
                if reward == 0:
                    # simulation A VS B
                    rewardA,rewardB = get_reward(agentA,agentB,Turn,env)
                    if Turn == A : reward = rewardA
                    if Turn == B : reward = rewardB

                # save in ReplayMemory
                if Turn == A : agentA.append_sample(state, action, reward, next_state, done)
                if Turn == B : agentB.append_sample(state, action, reward, next_state, done)

                env.__changeBoardTurn__(env.Board)

                if Turn == A : scoreA += reward
                if Turn == B : scoreB += reward


            # train each step

            if Turn == A and len(agentA.memory) > agentA.train_start:
                agentA.train_model()
            if Turn == B and len(agentB.memory) > agentB.train_start:
                agentB.train_model()

            if done:
                if Turn == A :
                    agentA.update_target_model()
                    scoresA.append(scoreA)
                    episodesA.append(eA)
                    eA += 1
                if Turn == B :
                    agentB.update_target_model()
                    scoresB.append(scoreB)
                    episodesB.append(eB)
                    eB += 1

                pylab.plot(episodesA,scoresA,'b')
                pylab.savefig("./save_graph/tictactoe_dqnA.png")

                pylab.plot(episodesB, scoresB, 'b')
                pylab.savefig("./save_graph/tictactoe_dqnB.png")

                print("episode:",e," score A:",scoreA," score B:",scoreB,
                      " epsilonA:",agentA.epsilon," epsilonB:",agentB.epsilon)

            Turn *= -1

    agentA.model.save_weights("./save_model/tictactoeA_dqn.h5")
    agentB.model.save_weights("./save_model/tictactoeB_dqn.h5")


