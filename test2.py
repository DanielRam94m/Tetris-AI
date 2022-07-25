#import argparse
#from turtle import width
import torch
#import cv2
from src.tetris import Tetris

#TODO: correr N veces cada modelo
#TODO: crear archivo con métricas obtenidas (modelo_, corrida_N, destroyed_lines, score)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.cuda.manual_seed(151)
else:
    torch.manual_seed(151)

def test():
    if device == 'cuda':
        model = torch.load("{}/tetris_2000".format('trained_models'))
    else:
        model = torch.load("{}/tetris_2000".format('trained_models'), map_location=lambda storage, loc: storage)
    model.eval()
    model.to(device) 
    env = Tetris()
    env.reset()
 
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        next_states = next_states.to(device)
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.perform_action(action)

        if done:
            break


if __name__ == "__main__":

    test()