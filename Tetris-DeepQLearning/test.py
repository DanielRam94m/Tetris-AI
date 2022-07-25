import torch
from tetris import Tetris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import device

def test(model_name, t):
    if device == 'cuda':
        model = torch.load("{}/{}".format('trained_models', model_name))
    else:
        model = torch.load("{}/{}".format('trained_models', model_name), map_location=lambda storage, loc: storage)
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
            return env.lines_destroyed, env.score


if __name__ == "__main__":
    print('\n-------------------------------------------------------------------------')
    times = int(input('digite cuantas veces quiere correr cada uno de los 5 modelos:  '))
    models = ['tetris_500', 'tetris_1000', 'tetris_1500', 'tetris_2000', 'tetris_2500']

    df_test_metrics = pd.DataFrame(columns=['model', 'run', 'lines destroyed', 'score'])

    # corremos el test n veces para cada uno
    for model in models:
        for t in range(times):
            print(f'running model {model}\t\ttime:{t+1}')
            lines_destroyed, score = test(model, t)
            # vamos llenando dataframe
            new_row = {'model':model, 'run':t, 'lines destroyed':lines_destroyed, 'score':score}
            df_test_metrics = df_test_metrics.append(new_row, ignore_index=True)

    # Generemos archivo csv con las métricas de test
    df_test_metrics.to_csv('generated_metrics/test_metrics.csv')

    # Se genera imagen de boxplot
    sns.set_style('whitegrid')
    sns.boxenplot(x='model', y='score', data=df_test_metrics)
    plt.savefig('generated_metrics/test_metrics.png')

    

