import numpy as np
import torch
import torch.nn as nn
from deep_q_network import DeepQNetwork
from collections import deque
from random import random, randint, sample

#Global variable device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.cuda.manual_seed(151)
else:
    torch.manual_seed(151)

class Agent():
    def __init__(self, epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma, epsilon, decay, batch_size):
        self.epochs = epochs
        self.batch_memory_size = batch_memory_size
        self.num_decay_epochs = num_decay_epochs
        self.model_save_interval = model_save_interval
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.batch_size = batch_size
        self.model = DeepQNetwork()
        self.replay_memory = deque(maxlen=self.batch_memory_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_model = nn.MSELoss()
    
    def learn(self):
        if len(self.replay_memory) < self.batch_memory_size / 10:
            return False
        
        batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
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

        y_batch = torch.cat(tuple(reward if done else reward + self.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        self.optimizer.zero_grad()
        loss = self.loss_model(q_values, y_batch)
        loss.backward()
        self.optimizer.step()
        return True


    def step(self, env, actual_state, ep):
        ''' Obtiene todos los posibles siguientes estados del tablero para la figura actual'''
        next_steps = env.get_next_states() 
        next_actions, next_states = zip(*next_steps.items()) # separo los estados de los IDs
        next_states = torch.stack(next_states) # Uno los tensores de cada estado en un solo tensor
        next_states = next_states.to(device)  
        with torch.no_grad():
            predictions = self.model(next_states)[:, 0]
        self.model.train()
        '''Disminucion gradual del epsilon'''
        epsilon = self.decay + (max(self.num_decay_epochs - ep, 0) * (
                self.epsilon - self.decay) / self.num_decay_epochs)
        ''' Escoge accion'''
        if random() > epsilon: # si variable aleatoria menor que valor epsilon
            # toma la mejor accion conocida
            index_action = torch.argmax(predictions).item()
        else:
            # elige una accion al azar 
            index_action = randint(0, len(next_steps) - 1)        
        action = next_actions[index_action]
        next_state = next_states[index_action, :]
        next_state = next_state.to(device) 
        ''' Ejecuta accion'''
        reward, done = env.perform_action(action)
        ''' Almacena en memoria '''
        self.replay_memory.append([actual_state, reward, next_state, done])

        return done, next_state
