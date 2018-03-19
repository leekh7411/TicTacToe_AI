import sys
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

class Agent_A2C:
    def __init__(self,state_size,action_size,load_model=False):
        self.load_model = load_model
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # actor-critic hyper-parameters
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # Make Policy-Net and Value-Net
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()


    # Actor is Policy network
    # get state and return action probability
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24,
                        input_dim=self.state_size,
                        activation="relu",
                        kernel_initializer="he_uniform"))
        actor.add(Dense(self.action_size,
                        activation='softmax',
                        kernel_initializer="he_uniform"))
        actor.summary()
        return actor

    # Critic is Value network
    # get state and return state's value
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24,
                         input_dim=self.state_size,
                         activation="relu",
                         kernel_initializer="he_uniform"))
        critic.add(Dense(24,
                         activation="relu",
                         kernel_initializer="he_uniform"))
        critic.add(Dense(self.value_size,
                         activation='linear',
                         kernel_initializer="he_uniform"))
        critic.summary()
        return critic

    # Choice action stochastically in Policy-Net's output
    def get_action(self,state):
        policy = self.actor.predict(state,batch_size=1).flatten()
        for i in range(state[0].__len__()):
            if state[0][i] != 0:
                policy[0][i] = 0
        return np.random.choice(self.action_size,1,p=policy)[0]

    # Update Policy-Net
    def actor_optimizer(self):
        action = K.placeholder(shape=[None,self.action_size])
        advantage = K.placeholder(shape=[None,])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entrophy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entrophy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights,[],loss)
        train = K.function([self.actor.input,action,advantage],[],
                           updates=updates)
        return train

    # Update Value-Net
    def critic_optimizer(self):
        target = K.placeholder(shape=[None,])
        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights,[],loss)
        train = K.function([self.critic.input,target],[],updates=updates)

        return train

    # Train every time-step
    def train_model(self,state,action,reward,next_state,done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1,self.action_size])
        act[0][action] = 1

        # Get 'Advantage' and 'Update Target'
        # using Bellman Expectation Equation
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor*next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updater([state, act, advantage])
        self.critic_updater([state,target])



