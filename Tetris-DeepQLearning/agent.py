import numpy as np
import torch
import torch.nn as nn
from deep_q_network import DeepQNetwork
from collections import deque
from random import random, randint, sample
from utils import device
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
    
    '''
    método que ejecuta algoritmo Q Learning para entrenar la red
    recibe un objeto de la clase Tetris y el estado actual
    retorna bool para saber si esta en modo aprendizaje
    '''
    def learn(self):
        #empieza el aprendizaje una vez que la memoria este llena
        if len(self.replay_memory) < self.batch_memory_size / 10:
            return False
        #Toma una muestra de la memoria (estado_actual, recompensa, siguiente estado, game_over)
        batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state.to(device) for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device) 
        next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)  
        #Calcula los valores q del estado actual
        q_values = self.model(state_batch)
        self.model.eval()
        with torch.no_grad():
            #Calcula los valores q del siguiente estado
            next_prediction_batch = self.model(next_state_batch)
        self.model.train()
        #Obtiene los valores esstimados (Target values)
        y_batch = torch.cat(tuple(reward if done else reward + self.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
        self.optimizer.zero_grad()
        loss = self.loss_model(q_values, y_batch)
        loss.backward()
        self.optimizer.step()
        return True

    '''
    método que ejecuta un step del agente en la simulacion  
    recibe un objeto de la clase Tetris y el estado actual
    retorna bool si el juego esta terminado y el siguiente estado
    '''
    def step(self, env, actual_state, ep):
        ''' Obtiene todos los posibles siguientes estados del tablero para la figura actual'''
        next_steps = env.get_next_states() 
        next_actions, next_states = zip(*next_steps.items()) # separo los estados de los IDs
        next_states = torch.stack(next_states).to(device)   # Uno los tensores de cada estado en un solo tensor
        #next_states = next_states.to(device) 
        '''Forward modelo''' 
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
        reward, game_over = env.perform_action(action)
        ''' Almacena en memoria (estado_actual, recompensa, siguiente estado, game_over)'''
        self.replay_memory.append([actual_state, reward, next_state, game_over])

        return game_over, next_state
