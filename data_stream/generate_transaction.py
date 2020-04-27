import pandas as pd
from random import randint

import os

import time


transactions = pd.read_csv('stream.csv', header='infer', sep=',')

output_directory = 'output'

i = 0

while True:
    # number of simultaneous transactions
    simultaneous = randint(1, 150)
    subset = transactions.iloc[i:i+simultaneous]

    file_name = 'transactions_{}_{}.csv'.format(simultaneous, str(time.time()).split('.')[0])
    path = os.path.join(output_directory, file_name)
    print('Exporting:', path)

    subset.to_csv(path, header=False, index=False)

    # waiting for more transactions    
    time.sleep(2)

    i = i + simultaneous if len(transactions) > i + simultaneous else 0
