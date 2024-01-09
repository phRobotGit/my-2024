#%%
from dataclasses import dataclass
from typing import Optional
import pandas as pd 
import numpy as np
import pypfopt 
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
# from pypfopt.base_optimizer import portfolio_performance
# from module.portfolio.optimizer import PortfolioOptimizer

@dataclass
class Portfolio():
    component_weight_matrix_df: pd.DataFrame
    '''
        "日期":pd.Timestamp,
        "weight_c1": float,
    '''
    
    component_price_matix_df: pd.DataFrame
    '''
        "日期":pd.Timestamp,
        "price_c1": float,
    '''
    
    component_return_matix_df: pd.DataFrame = None
    '''
        "日期":pd.Timestamp,
        "return_c1": float,
    '''
    
    portfolio_df:  Optional[pd.DataFrame] = None
    '''
        "日期": pd.Timestamp,
        "AUM": float,
        "return": float,
    '''
    
    _all_return_matix_df: pd.DataFrame = None
    '''
        "日期": pd.Timestamp,
        "return_p": float,
        "return_c1": float,
    '''
    
    _all_price_matix_df: pd.DataFrame = None
    '''
        "日期": pd.Timestamp,
        "price_p": float,
        "price_c1": float,
    '''
    
    # effrontier:pypfopt.EfficientFrontier = None  
    
    def __post_init__(self):
        W = self.component_weight_matrix_df.copy()
        P = self.component_price_matix_df.copy()
        
        # portfolio_report_df
        aum_seies = (W * P).sum(axis=1)
        return_series = np.log(aum_seies/aum_seies.shift())
        self.portfolio_df = pd.DataFrame({
            "aum": aum_seies.values,
            "return": return_series.values,
        }, index=aum_seies.index)
        
        # component_return_matix_df
        self.component_return_matix_df = np.log(P/P.shift()) 
        
        # _all_return_matix_df
        df_1 = self.component_return_matix_df.copy()
        df_2 = self.portfolio_df[["return"]].copy()
        self._all_return_matix_df = pd.merge(df_1, 
                 df_2.rename(columns={"return":"portfolio"}), 
                 left_index=True, 
                 right_index=True, how="left")

        # _all_price_matix_df
        df_1 = self.component_price_matix_df.copy()
        df_2 = self.portfolio_df[["aum"]].copy()
        self._all_price_matix_df = pd.merge(df_1, 
                 df_2.rename(columns={"aum":"portfolio_aum"}), 
                 left_index=True, 
                 right_index=True, how="left")
        
        
