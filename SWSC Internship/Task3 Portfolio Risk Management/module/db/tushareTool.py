#%%
import pandas as pd 
import numpy as np
import tushare as ts 


class ConnetorTuShare():
    token = None
    pro = None 
    
    def __init__(self, token):
        self.token = token 
        self.pro = ts.pro_api(token)

    
    def get_data(self, code_list, start_date, end_date):    
        code_str = ",".join(code_list)
        df_stock = self.pro.query('daily', 
                    ts_code=code_str, #'600519.SH, 300750.SZ, 513500.SZ' 
                    start_date=start_date,
                    end_date=end_date,)
        df_fund_list = []
        for code in code_list:
            df = self.pro.fund_daily(
                        ts_code=code, 
                        start_date=start_date, 
                        end_date=end_date)
            df_fund_list.append(df)
        df_fund = pd.concat(df_fund_list)
        df = pd.concat([df_stock, df_fund])
        return(df)