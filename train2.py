import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
#from src.tetris import Tetris2
from collections import deque


#todo: check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.cuda.manual_seed(151)
else:
    torch.manual_seed(151)

'''

'''
def train(epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma,
          initial_epsilon, final_epsilon, batch_size, width, height, block_size):
    ''' 
    #todo: check
    if torch.cuda.is_available():
        torch.cuda.manual_seed(151)  ## número primo que escogimos ##
    else:
        torch.manual_seed(151)
    '''
    env = Tetris(width=width, height=height, block_size=block_size)
    #env = Tetris2(width=width, height=height, block_size=block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    state = env.reset()
    '''
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()
    '''
    model.to(device)  #todo: check
    state = state.to(device)  #todo: check
    
        

    replay_memory = deque(maxlen=batch_memory_size)

    ep = 0
    while ep < epochs:
        next_steps = env.get_next_states()

        epsilon = final_epsilon + (max(num_decay_epochs - ep, 0) * (
                initial_epsilon - final_epsilon) / num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        '''
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        '''
        next_states = next_states.to(device)  #todo: check
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=True)

        '''
        if torch.cuda.is_available():
            next_state = next_state.cuda()
        '''
        next_state = next_state.to(device)  #todo: check
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            '''
            if torch.cuda.is_available():
                state = state.cuda()
            '''
            state = state.to(device)  #todo: check
        else:
            state = next_state
            continue
        if len(replay_memory) < batch_memory_size / 10:
            continue
        
        batch = sample(replay_memory, min(len(replay_memory), batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        '''
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        '''
        state_batch = state_batch.to(device)  #todo: check
        reward_batch = reward_batch.to(device)  #todo: check
        next_state_batch = next_state_batch.to(device)  #todo: check

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print('[Epoch {}/{}]\t\t[Lines destroyed: {}]\t\t[Score: {}]'.format(ep+1, epochs,
            final_cleared_lines, final_score))

        ep += 1
        if ep % model_save_interval == 0 and ep > 0:
            torch.save(model, f'trained_models/tetris_{ep}')


if __name__ == "__main__":
    epochs = 10                ## total de épocas por correr ##
    batch_memory_size = 3000   ##  ## 
    num_decay_epochs = 2000    ##  ## decay_epochs
    model_save_interval = 5  ## cada cuanto se salvará un modelo ##
    lr = 1e-3                  ## rango de aprendizaje ##
    gamma = 0.99               ##  ##
    initial_epsilon = 1        ##  ##
    final_epsilon = 1e-3       ##  ##
    batch_size = 512           ##  ##
    width = 10                 ## ancho en bloques del tetris ##
    height = 20                ## anlto en bloques del tetris ##
    block_size = 30            ## tamaño del bloque ##

    train(epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma,
          initial_epsilon, final_epsilon, batch_size, width, height, block_size)
