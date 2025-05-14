# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from stable_baselines3.common.logger import configure
from stable_baselines3 import A2C, PPO, DDPG
from data.data import ForexData
import yfinance as yf
import pandas as pd
import itertools
import sys
from constants import *

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)

# For visual styling of plots
plt.style.use('seaborn-v0_8')

path = "C:\\Users\\rober\\TUD-CSE-RP-RLinFinance\\data\\forex\\EURUSD\\15M\\BID\\10.05.2022T00.00-09.05.2025T20.45.csv"
data = ForexData(path)
df = data.df

print(df.head())
