import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Generate Result Lottery")


#load data 
powerMegaPath=  os.path.dirname(os.path.abspath(__file__))+'/results/mega.csv'

data = pd.read_csv(powerMegaPath)

logger.info('Loaded data')

# Assume 'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'NumS' are columns in the CSV file.

columns = ["Num1","Num2","Num3","Num4","Num5","NumS"]
max_white=45
max_red=44

# Initialize weights for numbers 1-44/45 for Num1-Num5 and 1-44/45 for NumS

weights = np.ones(max_white +1)
weights_s =np.ones(max_red + 1)

def update_weights(row):
    for col in columns[:1] # Update weights for Num1-Num5



