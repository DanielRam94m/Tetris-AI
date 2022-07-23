import argparse
from turtle import width
import torch
import cv2
from src.tetris import Tetris



def test(width=10, height=20, block_size=30, fps=300, output='output.mp4v'):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/tetris_2000".format('trained_models'))
    else:
        model = torch.load("{}/tetris_2000".format('trained_models'), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=width, height=height, block_size=block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), fps,
                          (int(1.5*width*block_size), height*block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break
        


if __name__ == "__main__":
    width = 10
    height = 20
    block_size = 30
    fps = 300
    output = 'output.mp4v'

    test(width, height, block_size, fps, output)

