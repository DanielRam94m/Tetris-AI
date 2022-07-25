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
from utils import device

'''
método para entrenar el modelo 
recibe los parametros del modelo y cada cuantas iteraciones tiene que pasar para guardar el modelo
retorna los modelos de cada n iteraciones
'''
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
        game_over = False
        learn = False
        #Mientras no se acaba el juego
        while not game_over:
            '''SIMULATION STEP'''
            game_over, next_state = agent.step(env, actual_state, ep)
            actual_state = next_state.to(device)  
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
    print('\n----------------------------------------------')
    ## total de épocas por correr ##
    epochs = int(input('digite cantidad de épocas: '))
    ## cada cuanto se salvará un modelo ##
    model_save_interval = int(input('digite intervalo para salvar modelo: '))
    batch_memory_size = 3000   ## tamaño de la memoria ## 
    num_decay_epochs = 2000    ## decay_epochs 
    lr = 1e-3                  ## rango de tasa aprendizaje ##
    gamma = 0.99               ## gamma
    epsilon = 1                ## epsilon
    decay = 1e-3               ## decay
    batch_size = 512           ## tamaño de memoria para entrenamiento

    train(epochs, batch_memory_size, num_decay_epochs,
          model_save_interval, lr, gamma, epsilon, 
          decay, batch_size)
