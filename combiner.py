import numpy as np
import pandas as pd
import os

def string_fixer(word):
    new_word = word.replace(' ', '_')
    new_word = new_word.replace('/', '_')
    return new_word



def main_function():
    owd = os.getcwd()
    # Input location of pre-processed dataset
    file_path = '2014-2015_raw.csv'
    df = pd.read_csv(file_path)

    list_of_causes = list(set(df['STAT_CAUSE_DESCR']))
    folders_list = [string_fixer(n) for n in list_of_causes]

    new_np = []
    i=1
    for item in folders_list:
        for type in ['train']:
            item_array = np.load('Data/'+str(item)+'/'+str(type)+'/'+str(item)+'_'+str(type)+'.npy')
        if i==1:
            new_np = item_array
            i=0
            print('Added '+str(item))
        else:
            new_np = np.append(new_np,item_array)
            print('Added ' + str(item))

    np.save('Full_Array', new_np)

if __name__ == '__main__':
    main_function()
