
# 优化投资组合
from dataclasses import dataclass
import pathlib
from pickletools import pyfloat

import sys
from typing import Any
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
import matplotlib.pyplot as plt
# sys

# @dataclass
class PortfolioOptimizer():
    portfolio:Portfolio = None 
    effrontier:pypfopt.EfficientFrontier = None  

    def __init__(self, portfolio):
        P = portfolio.component_price_matix_df.copy()
        W = portfolio.component_weight_matrix_df.copy()
        
        mu = expected_returns.mean_historical_return(P) # input price data
        S = risk_models.sample_cov(P)
        
        ef = EfficientFrontier(mu, S)
        
        self.effrontier = ef.deepcopy()
        self.portfolio = portfolio
        

    def plot_effrontier(self):
        fig, ax = plt.subplots()
        ef = self.effrontier.deepcopy()
        plotting.plot_efficient_frontier(ef, ax=ax, 
                                         show_assets=False, 
                                        #  show_tickers=True,
                                         points=1000)

        # Find the tangency portfolio
        ef_max_sharpe = self.effrontier.deepcopy()
        ef_max_sharpe.max_sharpe()
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

        # Generate random portfolios
        n_samples = 12000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

        # Output
        ax.set_title("Efficient Frontier with random portfolios")
        ax.legend()
        plt.tight_layout()
        # plt.show()
        
        # 增加散点
        asset_mu = ef.expected_returns
        asset_sigma = np.sqrt(np.diag(ef.cov_matrix))

        ax.scatter(
                    asset_sigma,
                    asset_mu,
                    s=30,
                    color="k",
                    label="assets",
                )
        
        for i, label in enumerate(ef.tickers):
            ax.annotate(label, (asset_sigma[i], asset_mu[i]-0.04),
                        ha='center',
                        fontsize = 6)
        # 增加mmv
        ef_mmv = self.effrontier.deepcopy()
        ef_mmv.min_volatility()
        ret, vol, _ = ef_mmv.portfolio_performance()
        x_mmv = vol  # 示例 x 坐标
        y_mmv = ret  # 示例 y 坐标
        ax.scatter(x_mmv, y_mmv, marker="o", s=100, c="blue", label="mmv")
        # ax.annotate("mmv", (x_mmv, y_mmv), ha='center', fontsize=10)
        plt.legend()

        
        return(fig)
    
    
    
    
    
    def generate_opt_portfolio(self, type):
        W = self.portfolio.component_weight_matrix_df.copy()
        if type == "max_sharpe":
            ef = self.effrontier.deepcopy()
            w = ef.max_sharpe()
        if type == "mmv":
            ef = self.effrontier.deepcopy()
            w = ef.min_volatility()
        W_opt = pd.DataFrame({k:[v]*W.shape[0] for k,v in w.items()},
                             index = W.index)
        p = Portfolio(component_price_matix_df=self.portfolio.component_price_matix_df,
                  component_weight_matrix_df=W_opt,)
        return(p)

