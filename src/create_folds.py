from sklearn import model_selection
import pandas as pd
import numpy as np
import os

def main():

    # read the data 
    data = pd.read_csv(os.path.join('..', 'input', 'train.csv'))

    print(f'Length of the original dataset {len(data)}')

    # data augumention if ( A->B then B!->A)
    data_label1 = data[data['label'] == 1]
    data_label0 = data[data['label'] == 0]

    data_label_change = data_label1.rename(columns={'pre requisite': 'concept', 'concept': 'pre requisite'}, inplace=False)

    data_label0 = data_label0.append(data_label_change).drop_duplicates(subset=['concept', 'pre requisite'])
    
    data_label0['label'] = 0 # change the label for the agumentation
    
    data = pd.concat([data_label0, data_label1])

    print(f'Length of the augumented dataset {len(data)}')

    # seperate into folds

    data['kfold'] = -1

    data = data.sample(frac = 1).reset_index(drop = True)

    y = data.label.values

    kf = model_selection.StratifiedGroupKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X = data, y = y)):
        data.loc[v_, 'kfold'] = f

    data.to_csv(os.path.join('..', 'input', 'train_folds.csv'), index = False)

if __name__ == '__main__':
    main()



