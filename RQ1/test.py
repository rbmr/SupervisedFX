# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from stable_baselines3.common.logger import configure
from stable_baselines3 import A2C, PPO, DDPG
import yfinance as yf
import pandas as pd
import itertools

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)

# For visual styling of plots
plt.style.use('seaborn-v0_8')

import yfinance as yf
import pandas as pd
