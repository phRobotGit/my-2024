#%%
import pandas as pd 
import numpy as np 
import pathlib
# from module.portfolio.portfolio import Portfolio
from module.portfolio.optimizer import PortfolioOptimizer
from module.portfolio.comparer import PortfolioShow
from module.risk.indicator import calculate_indicator_price
from module.visz.plot import plot_price_matrix
from module.report.reporter import ReporterDP
from module.db.tushareTool import ConnetorTuShare
from pypfopt import plotting

# config
START_DATE = "20230101" 
END_DATE = "20231229"
TUSHARE_TOKEN = r"854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c"


# 获取交易日历
import tushare as ts 
pro = ts.pro_api(TUSHARE_TOKEN)
calendar_series = pro.query('daily', 
               ts_code='600519.SH', 
               start_date=START_DATE,
               end_date=END_DATE)["trade_date"]
calendar_series = [pd.to_datetime(i,format="%Y%m%d") for i in calendar_series]
date_list = [i for i in pd.date_range(START_DATE, END_DATE) if i in calendar_series]

# 读入持仓数据
df_holding = pd.read_excel(pathlib.Path(r"src/持仓数据20231229.xlsx"),
                   skiprows=1,dtype={1: str})
df_holding["证券分类"].ffill(inplace=True)
def f(code):
    prefix = str(code)[:2]
    prefix_suffux_dict = {
        "60": ".SH", 
        "68": ".SH",
        "30": ".SZ",
        "00": ".SZ", 
        "01": ".SH",
        "51": ".SH", 
        "15": ".SZ"}
    return(str(code)+prefix_suffux_dict[prefix])
df_holding["tushare_code"] = df_holding["证券代码"].apply(f)

# 制作持仓矩阵
import tushare as ts 
pro = ts.pro_api(TUSHARE_TOKEN)


df_list = []
w = (df_holding["持仓数量"] * df_holding['成本价'])
w = w/w.sum() 
w = w.tolist() 
for date in date_list:
    df = pd.DataFrame(
        {date:w},
        index = df_holding["tushare_code"].tolist(), 
    ).T
    df_list.append(df)
component_weight_matrix_df =  pd.concat(df_list)


# 制作成分价格矩阵
import tushare as ts 
pro = ts.pro_api(TUSHARE_TOKEN)
tsCon = ConnetorTuShare(TUSHARE_TOKEN)
df_tushare = tsCon.get_data(
                    code_list=df_holding["tushare_code"].tolist(),
                    start_date=START_DATE, 
                    end_date=END_DATE)
df_tushare["trade_date"] = pd.to_datetime(df_tushare["trade_date"],format="%Y%m%d")
print(df_tushare["ts_code"].value_counts())


def f(df):
    '''增加reutrn'''
    df.sort_values(by="trade_date",inplace=True, ascending=True)
    df["return"] = np.log(df["close"]/df["close"].shift())
    return(df)
df_tushare = df_tushare.groupby("ts_code").apply(f).reset_index(drop=True)
component_price_matrix_df = df_tushare.pivot(columns="ts_code", values="close", index="trade_date")

component_price_matrix_df.to_csv(
    pathlib.Path("src/input/price_1y_daily.csv"),
    # index=False
    )

component_weight_matrix_df.to_csv(
    pathlib.Path("src/input/weight_1y_daily.csv"),
    # index=False
)


