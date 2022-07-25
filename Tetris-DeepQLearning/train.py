import numpy as np
import torch
import torch.nn as nn
from random import random, randint, sample
from deep_q_network import DeepQNetwork
from agent import Agent
from tetris import Tetris
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Global variable device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.cuda.manual_seed(151)
else:
    torch.manual_seed(151)


def train(epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma,
          epsilon, decay, batch_size):
    # Creamos una instancia del agente
    agent = Agent(epochs, batch_memory_size, num_decay_epochs, model_save_interval, lr, gamma, 
                  epsilon, decay, batch_size)
    # Creamos dataframe de métricas de test
    df_train_metrics = pd.DataFrame(columns=['epoch', 'lines destroyed', 'score'])
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
            '''SIMULATION STEP'''
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
            new_row = {'epoch':ep, 'lines destroyed':final_lines_destroyed, 'score':final_score}
            df_train_metrics = df_train_metrics.append(new_row, ignore_index=True)
        else:
            print('[Loading simulation in memory...]')
    # Generemos archivo csv con las métricas de train
    df_train_metrics.to_csv('generated_metrics/train_metrics.csv')

    # Se genera imagen de lineplot
    sns.set_style('whitegrid')
    #sns.boxenplot(x='epoch', y='score', data=df_train_metrics)
    sns.lineplot(x='epoch', y='score', data=df_train_metrics)
    plt.savefig('generated_metrics/train_metrics.png')


if __name__ == "__main__":
    epochs = 2500              ## total de épocas por correr ##
    batch_memory_size = 3000   ## tamaño de la memoria ## 
    num_decay_epochs = 2000    ##  ## 
    model_save_interval = 500  ## cada cuanto se salvará un modelo ##
    lr = 1e-3                  ## rango de aprendizaje ##
    gamma = 0.99               ##  ##
    epsilon = 1                ##  ##
    decay = 1e-3               ##  ##
    batch_size = 512           ##  ##

    train(epochs, batch_memory_size, num_decay_epochs,
          model_save_interval, lr, gamma, epsilon, 
          decay, batch_size)
