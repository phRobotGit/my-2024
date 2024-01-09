
# 优化投资组合
from dataclasses import dataclass
import pathlib
from pickletools import pyfloat

import sys
from typing import Any, Dict
import pypfopt
sys.path.append(pathlib.Path(__file__).parent.parent.parent)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt 
import pandas as pd 
from module.portfolio.constructor import Portfolio
import matplotlib.pyplot as plt 
import numpy as np


def PortfolioShow(): 
    portfolio_dict:Dict = None 
    
    def __init__(self, portfolio_dict):
        self.portfolio_dict = portfolio_dict
    
    def compare_summarization():
        pass 
    
    def compare_report():
        pass 