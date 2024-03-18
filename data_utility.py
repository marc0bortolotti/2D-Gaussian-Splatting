import os
import pandas as pd

dataframe = pd.DataFrame(columns=['folder', 'num_splitted', 'num_pruned', 'loss'])


with open('output.txt', 'w') as output:
    for folder in os.listdir('test'):
        num_splitted = 0
        num_pruined = 0
        with open('test/' + folder + '/statistics.txt') as f:
            print(folder)
            lines = f.readlines()
            for line in lines:
                if 'Number of splitted points' in line:
                    num_splitted += int(line.split(': ')[1])
                elif 'number of pruned points' in line:
                    num_pruined += int(line.split(': ')[1])
                elif 'Loss' in line:
                    loss = float(line.split(': ')[1])

        dataframe = dataframe.append({'folder': folder, 'num_splitted': num_splitted, 'num_pruned': num_pruined, 'loss': loss}, ignore_index=True)

    output.write(dataframe.to_string())
            