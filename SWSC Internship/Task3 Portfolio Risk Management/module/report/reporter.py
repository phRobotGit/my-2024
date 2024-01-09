from dataclasses import dataclass
from typing import Optional
import datapane as dp
from matplotlib.dviread import Page
from numpy import block
import pandas as pd 


class ReporterDP():
    view: Optional[dp.View] = None

    # portfolio_mmv = portfolio_optimizer.generate_opt_portfolio(type="mmv")
    # portfolio_mmv.performance = Performancer(portfolio_mmv).performance()
    def __init__(self, 
                    df_indicator, 
                    fig_price_line,
                    df_holding,
                    fig_eff,
                    portfolio,
                    portfolio_mmv,
                    portfolio_maxSharp):
        
        w1 = portfolio.component_weight_matrix_df[[i for i in df_holding["code"]]].iloc[-1,:]
        aum1 = portfolio.portfolio_df["aum"]
        
        w2 = portfolio_maxSharp.component_weight_matrix_df[[i for i in df_holding["code"]]].iloc[-1,:]
        aum2 = portfolio_maxSharp.portfolio_df["aum"]
        
        w3 = portfolio_mmv.component_weight_matrix_df[[i for i in df_holding["code"]]].iloc[-1,:]
        aum3 = portfolio_mmv.portfolio_df["aum"]        

        self.view = dp.View(
            dp.Page(
                title="Components Page",
                blocks = [
                    dp.DataTable(df_holding, label="Data"),
                    dp.Plot(fig_price_line)
                ]
            ),
            dp.Page(
                title="Portfolio Page",
                blocks = [
                    dp.Select(blocks=[
                        dp.Table(w1, label="w1"),
                        dp.Table(w2, label="w2"),
                        dp.Table(w3, label="w3"),
                    ])
                ]
            )
        ) 
    
    def save_to(self, save_path):
        dp.enable_logging()
        dp.save_report(self.view, 
                       path = save_path, 
                       open=False)
        