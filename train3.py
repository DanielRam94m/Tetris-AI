import numpy as np
import torch
import torch.nn as nn
from random import random, randint, sample
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque

#Global variable device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.cuda.manual_seed(151)
else:
    torch.manual_seed(151)

class Agent():
    def __init__(self, epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma, initial_epsilon, final_epsilon, batch_size):
        self.epochs = epochs
        self.batch_memory_size = batch_memory_size
        self.num_decay_epochs = num_decay_epochs
        self.model_save_interval = model_save_interval
        self.lr = lr
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.model = DeepQNetwork()
        self.replay_memory = deque(maxlen=self.batch_memory_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_model = nn.MSELoss()
    
    def learn(self):
        if len(self.replay_memory) < self.batch_memory_size / 10:
            return False
        
        batch = sample(self.replay_memory, min(len(self.replay_memory), batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        state_batch = state_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        q_values = self.model(state_batch)
        self.model.eval()
        with torch.no_grad():
            next_prediction_batch = self.model(next_state_batch)
        self.model.train()

        y_batch = torch.cat(tuple(reward if done else reward + gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        self.optimizer.zero_grad()
        loss = self.loss_model(q_values, y_batch)
        loss.backward()
        self.optimizer.step()
        return True


    def step(self, env, actual_state, ep):
        next_steps = env.get_next_states()  ###cambiar a get state 

        epsilon = self.final_epsilon + (max(self.num_decay_epochs - ep, 0) * (
                self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states) ###get state
        next_states = next_states.to(device)  
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(next_states)[:, 0]
        self.model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        next_state = next_state.to(device) 

        action = next_actions[index]

        reward, done = env.perform_action(action)
        
        self.replay_memory.append([actual_state, reward, next_state, done])

        return done, next_state

def train(epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma,
          initial_epsilon, final_epsilon, batch_size):
    
    # Creamos una instancia del agente
    agent = Agent(epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma, initial_epsilon, final_epsilon, batch_size)
    # Creamos una instancia del juego
    env = Tetris()
    #  Se resetea el entorno 
    actual_state = env.reset()
    # Se asigna el modelo y estado a la GPU o CPU
    agent.model.to(device)  
    actual_state = actual_state.to(device)  
    '''SIMULATION'''
    ep = 0
    while ep < epochs:
        done = False
        learn = False
        #Mientras no se acaba el juego
        while not done:
            '''STEP'''
            done, next_state = agent.step(env, actual_state, ep)
            actual_state = next_state  
            actual_state = actual_state.to(device)
        learn = agent.learn()
        final_score = env.score
        final_lines_destroyed = env.lines_destroyed
        actual_state = env.reset()
        if learn:
            ep += 1
            if ep % model_save_interval == 0 and ep > 0:
                torch.save(agent.model, f'trained_models/tetris_{ep}')
            print('[Epoch {}/{}]\t\t[Lines destroyed: {}]\t\t[Score: {}]'.format(ep, epochs, final_lines_destroyed, final_score))
        else:
            print('[Loading simulation in memory...]')


if __name__ == "__main__":
    epochs = 10                ## total de épocas por correr ##
    batch_memory_size = 3000   ##  ## 
    num_decay_epochs = 2000    ##  ## decay_epochs
    model_save_interval = 10    ## cada cuanto se salvará un modelo ##
    lr = 1e-3                  ## rango de aprendizaje ##
    gamma = 0.99               ##  ##
    initial_epsilon = 1        ##  ##
    final_epsilon = 1e-3       ##  ##
    batch_size = 512           ##  ##

    train(epochs, batch_memory_size, num_decay_epochs,
          model_save_interval, lr, gamma, initial_epsilon, 
          final_epsilon, batch_size)
