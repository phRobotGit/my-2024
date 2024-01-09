
# 优化投资组合
from dataclasses import dataclass
import pathlib
from pickletools import pyfloat

import sys
from typing import Any, Dict
import pypfopt
sys.path.append(pathlib.Path(__file__).parent.parent.parent)
from pypfopt.base_optimizer import portfolio_performance
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt 
import pandas as pd 
from module.portfolio.constructor import Portfolio
import matplotlib.pyplot as plt 
import numpy as np


class Performancer():
    portfolio:Portfolio = None 
    effrontier:pypfopt.EfficientFrontier = None  
    
    def __init__(self, portfolio:Portfolio):
        P = portfolio.component_price_matix_df.copy()
        W = portfolio.component_weight_matrix_df.copy()
        mu = expected_returns.mean_historical_return(P) # input price data
        S = risk_models.sample_cov(P)
        
        ef = EfficientFrontier(mu, S)
        
        self.effrontier = ef.deepcopy()
        self.portfolio = portfolio
    
    # @property
    def performance(self):
        ef = self.effrontier.deepcopy()
        
        performance_list = portfolio_performance(
            weights=self.portfolio.component_weight_matrix_df.iloc[-1,:],
            expected_returns=ef.expected_returns,
            cov_matrix=ef.cov_matrix,
            verbose=True,
            risk_free_rate=0.018
        )
        
        return(performance_list)
     