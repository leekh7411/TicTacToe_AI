import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

class Agent_DQN:
    def __init__(self,state_size,action_size,load_model=False):
        self.state_size = state_size
        self.action_size = action_size
        self.load_model = load_model

        # Hyper parameters
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 100

        # Replay memory
        self.memory = deque(maxlen=500)

        # initialize models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/tictactoeA_dqn.h5")

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_model_from(self,_model):
        self.target_model.set_weights(_model.get_weights())

    def build_model(self):
        model = Sequential()
        model.add(Dense(64,input_dim=self.state_size,activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size,activation='linear',kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self,state):
        # epsilon-greedy law
        if not self.load_model and np.random.rand() < self.epsilon:
            action_list = []
            for i in range(state[0].__len__()):
                if state[0][i] == 0:
                    action_list.append(i)
            action_idx = random.randrange(action_list.__len__())
            return action_list[action_idx]
        else:
            q_val = self.model.predict(state)
            for i in range(state[0].__len__()):
                if state[0][i] != 0:
                    q_val[0][i] = 0

            return np.argmax(q_val[0])

    def append_sample(self,state,action,reward,next_state,done):
        # Deep Q-Learning Based <S,A,R,S'>
        # save sample in replay memory
        self.memory.append((state,action,reward,next_state,done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory,self.batch_size)

        # [batch_size] X [state_size]
        states = np.zeros((self.batch_size,self.state_size))
        next_states = np.zeros((self.batch_size,self.state_size))
        actions, rewards, dones = [],[],[]

        for i in range(self.batch_size):
            # <S(0) , A(1) , R(2) , S'(3) , Done(4) >
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        '''
        All Q-Function about 'current states' and 'next states'
        In DQN, we divide models as 'For Update' and 'For Predict'
        self.model is the 'For Update'
        self.target_model is the 'For Predict'
        '''
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        '''
        Update target using Bellman Optimal Equation
        Q(s,a) <---- E[R + lr * max_val_in_all_actions(Q(s',a*))]
        'E' is expectation. Here we use NeuralNet for the 'E'
        '''

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(states,target,batch_size=self.batch_size,epochs=1,verbose=0)





















