import numpy as np
import pandas as pd
from spell_check import *

df = pd.read_csv('essays_dataset/index.csv', sep=';')

for filename in df['filename']:
    with open('essays_dataset/essays/' + filename, 'r') as file:
        essay = file.read()

        df.loc[df['filename'] == filename, 'essay'] = essay

df.to_csv('essays_dataset/preprocessed_index.csv', index=False)