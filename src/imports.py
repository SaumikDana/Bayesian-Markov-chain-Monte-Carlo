import setup_path

# General purpose libraries
import os
import re
import warnings
from itertools import combinations
from math import ceil
import pickle

# Numerical and data manipulation
import numpy as np
import pandas as pd
import cvxpy as cp

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Time series analysis
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from statsmodels.tsa.stattools import (
    acf,
    bds,
    coint,
    adfuller,
)
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

# Statistical analysis
from scipy.stats import (
    norm,
    chi2,
    kurtosis,
    pearsonr,
    spearmanr,
    kendalltau,
    shapiro
)
from scipy.interpolate import griddata
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.linalg import cholesky

# Machine learning
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import shap

# Financial and econometric modeling
from arch.unitroot import PhillipsPerron
from arch import arch_model
from numpy.random import multivariate_normal as mv_normal

# Forecasting and utilities
import yfinance as yf
import ta
import dask.dataframe as dd

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import random
from tqdm import trange
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

import numpy as np

import numpy as np
from scipy import integrate
from math import exp, log, sin

from scipy.stats import gaussian_kde
import time
