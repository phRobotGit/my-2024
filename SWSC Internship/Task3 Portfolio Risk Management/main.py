#%%
from xmlrpc.client import TRANSPORT_ERROR
from cycler import V
import pandas as pd 
import numpy as np 
import pathlib
import pypfopt
from module.portfolio.constructor import Portfolio
from module.portfolio.optimizer import PortfolioOptimizer
from module.portfolio.performancer import Performancer
from module.portfolio.comparer import PortfolioShow
from module.risk.indicator import calculate_indicator_price
from module.visz.plot import plot_price_matrix, plot_combination_chart
from module.report.reporter import ReporterDP
from module.db.tushareTool import ConnetorTuShare
from pypfopt import plotting

# config
START_DATE = "20230101" 
END_DATE = "20231229"
# END_DATE = "20230901"
TUSHARE_TOKEN = r"854634d420c0b6aea2907030279da881519909692cf56e6f35c4718c"
FLAG_BACKTEST = True

# Part(1): read data
# weight
component_weight_matrix_df = pd.read_csv(
    pathlib.Path("src/input/weight_1y_daily.csv"),
    index_col=0
)

window_1 = [pd.to_datetime(i) >= pd.to_datetime(START_DATE) and pd.to_datetime(i) <= pd.to_datetime(END_DATE) for i in component_weight_matrix_df.index]
component_weight_matrix_df = component_weight_matrix_df[window_1]

# price
component_price_matrix_df = pd.read_csv(
    pathlib.Path("src/input/price_1y_daily.csv"),
    index_col=0
)
window_2 = [pd.to_datetime(i) >= pd.to_datetime(START_DATE) and pd.to_datetime(i) <= pd.to_datetime(END_DATE) for i in component_price_matrix_df.index]
component_price_matrix_df = component_price_matrix_df[window_2]

# holding
df_holding = pd.read_excel(pathlib.Path(r"src/持仓数据20231229.xlsx"),
                   skiprows=1,dtype={1: str})
df_holding["证券分类"].ffill(inplace=True)
code_name_dict = dict(zip(df_holding["code"], df_holding["证券简称"]))




# Part(2): 制作portfolio
component_weight_matrix_df = component_weight_matrix_df[[col for col in component_price_matrix_df]] #简化

portfolio = Portfolio(component_weight_matrix_df=component_weight_matrix_df,
                      component_price_matix_df=component_price_matrix_df,
                      )

portfolio.performance = Performancer(portfolio).performance()
w1 = portfolio.component_weight_matrix_df[[i for i in df_holding["code"]]].iloc[-1,:]
aum1 = portfolio.portfolio_df["aum"]

# Part(3):优化portfolio
portfolio_optimizer = PortfolioOptimizer(portfolio=portfolio)


df_corr = portfolio.component_return_matix_df.copy()
df_corr = df_corr[[i for i in df_holding["code"]]]
# df.columns = [code_name_dict[i] for i in df.columns]
import seaborn as sns 
fig_corr_heat = sns.heatmap(
    df_corr.corr()
)


# 构建 max sharp 投资组合 
portfolio_maxSharp = portfolio_optimizer.generate_opt_portfolio(type="max_sharpe")
portfolio_maxSharp.performance = Performancer(portfolio_maxSharp).performance()

w2 = portfolio_maxSharp.component_weight_matrix_df[[i for i in df_holding["code"]]].iloc[-1,:]
aum2 = portfolio_maxSharp.portfolio_df["aum"]
# 构建 mmv 投资组合
portfolio_mmv = portfolio_optimizer.generate_opt_portfolio(type="mmv")
portfolio_mmv.performance = Performancer(portfolio_mmv).performance()
w3 = portfolio_mmv.component_weight_matrix_df[[i for i in df_holding["code"]]].iloc[-1,:]
aum3 = portfolio_mmv.portfolio_df["aum"]

# 输出权重
df = pd.DataFrame({
    "port": w1,
    "mas":w2,
    "mmv":w3
})
df.to_csv("src/weight.csv")


df = pd.DataFrame({
    "port": aum1,
    "mas":aum2,
    "mmv":aum3
})
df.to_csv("src/aum.csv")

# 画图
fig_eff = portfolio_optimizer.plot_effrontier()


# reporter部分
# 汇报 risk indicators
df_indicator = portfolio._all_price_matix_df.apply(calculate_indicator_price, axis=0)


# 画图
fig_price_line = plot_price_matrix(
                    df_price_matrix=portfolio._all_price_matix_df,
                    width_y_per = 3.3,  
                    width_x = 20,
                    save_path="src/plot_price_matrix.png")




# 增加回测模块
# config
# START_DATE = "20230101" 
if FLAG_BACKTEST:
    b_START_DATE = END_DATE
    b_END_DATE = "20240108" 
    
    # price
    b_P_df = pd.read_csv(
        pathlib.Path("src/input/price_1y_daily.csv"),
        index_col=0
    )
    window_2 = [pd.to_datetime(i) >= pd.to_datetime(b_START_DATE) and pd.to_datetime(i) <= pd.to_datetime(b_END_DATE) for i in b_P_df.index]
    b_P_df = b_P_df[window_2]

    df_list = []
    for index, row in b_P_df.iterrows():
        df = pd.DataFrame(w2).T
        # df.index= index
        df_list.append(df)
    b_W_df = pd.concat(df_list)
    b_W_df.index = b_P_df.index
    b_W_df = b_W_df[[i for i in b_P_df.columns]]
    portfolio_b1 = Portfolio(component_weight_matrix_df=b_W_df,
                      component_price_matix_df=b_P_df,
                      )
    portfolio_b1._all_price_matix_df.apply(calculate_indicator_price, axis=0)
    portfolio_b1.performance = Performancer(portfolio_b1).performance()
    w_b1 = portfolio_b1.component_weight_matrix_df[[i for i in df_holding["code"]]].iloc[-1,:]







# 制作报告
df_w1 = pd.DataFrame(w1)
df_w1.columns = ["weights"]
df_w2 = pd.DataFrame(w2)
df_w2.columns = ["weights"]
df_w3 = pd.DataFrame(w3)
df_w3.columns = ["weights"]

df_wb1 = pd.DataFrame(w_b1)
df_wb1.columns = ["weights"]

import matplotlib.pyplot as plt 
# 绘制饼图
p1_pie = plt.figure(figsize=(8, 8))
df = df_w1.copy()
plt.pie(df["weights"], labels=df.index, autopct='%1.1f%%')
plt.axis('equal')  # 设置饼图为圆形

import datapane as dp 
page_1 = dp.Page(title="投资标的（成分）",
                 blocks=[
                     dp.DataTable(df_holding, label="holding info"),
                     dp.Plot(fig_price_line, scale=0.5),
                     dp.Plot(fig_corr_heat),
                 ])
page_2 = dp.Page(title="投资组合",
                 blocks=[
                    dp.Text("原投资组合："),
                    dp.Group(blocks=[
                        dp.BigNumber(heading="期望年化收益率", value=f"{portfolio.performance[0]:.2%}"),
                        dp.BigNumber(heading="年化波动", value=f"{portfolio.performance[1]:.2%}"),
                        dp.BigNumber(heading="夏普比率", value=f"{portfolio.performance[2]:.2f}"),
                    ],columns =3),
                    dp.Plot(plot_combination_chart(portfolio.portfolio_df)),
                    dp.Plot(p1_pie),
                    
                    dp.Text("有效前沿与优化投资组合："),
                    fig_eff,
                    dp.Select(blocks=[
                        
                        dp.Group(blocks=[
                            dp.Group(blocks=[
                                dp.BigNumber(heading="期望年化收益率", value=f"{portfolio_maxSharp.performance[0]:.2%}"),
                                dp.BigNumber(heading="年化波动", value=f"{portfolio_maxSharp.performance[1]:.2%}"),
                                dp.BigNumber(heading="夏普比率", value=f"{portfolio_maxSharp.performance[2]:.2f}"),
                            ],columns =3),
                            dp.Plot(plot_combination_chart(portfolio_maxSharp.portfolio_df)),
                            dp.Text("成份"),
                            dp.Table(df_w1, label="成分"),  
                        ],label="最大夏普"),
             
                        dp.Group(blocks=[
                            dp.Group(blocks=[
                                dp.BigNumber(heading="期望年化收益率", value=f"{portfolio_mmv.performance[0]:.2%}"),
                                dp.BigNumber(heading="年化波动", value=f"{portfolio_mmv.performance[1]:.2%}"),
                                dp.BigNumber(heading="夏普比率", value=f"{portfolio_mmv.performance[2]:.2f}"),
                            ],columns =3),
                            dp.Plot(plot_combination_chart(portfolio_mmv.portfolio_df)),
                            dp.Text("成份"),
                            dp.Table(df_w2, label="成分"),
                            dp.Text("成份"),
                        ],label="最小方差"),
                        
                        dp.Group(blocks=[
                            dp.Text("开通融资融券后："),
                            dp.Text("进一步定制化风险，方差"),
                            # dp.Table(df_w1, label="w1"),
                        ],label="允许融资融券"),
                        
                        dp.Group(blocks=[
                            dp.Text("新"),
                            # dp.Text("进一步定制化风险，方差"),
                            # dp.Table(df_w1, label="w1"),
                        ],label="特定风险敞口要求"),
                    ])
                 ])

page_3 = dp.Page(title="回测",
                 blocks = [
                    dp.Text("回测框架"),
                    dp.Select(
                        blocks=[
                            dp.Group(blocks=[
                                dp.Text("时间段一"),
                                dp.Text("时间段二")
                            ],label="回测方案一"),
                            dp.Group(blocks=[
                                dp.Group(blocks=[
                                    dp.BigNumber(heading="期望年化收益率", value=f"{portfolio_b1.performance[0]:.2%}"),
                                    dp.BigNumber(heading="年化波动", value=f"{portfolio_b1.performance[1]:.2%}"),
                                    dp.BigNumber(heading="夏普比率", value=f"{portfolio_b1.performance[2]:.2f}"),
                                ],columns =3),
                                dp.Plot(plot_combination_chart(portfolio_b1.portfolio_df)),
                                dp.Text("成份"),
                                dp.Table(df_wb1, label="成分"),
                                dp.Text("成份"),
                            ],label="最大夏普回测 1-9, 9-12"),
                            # dp.Group(blocks=[
                            #     dp.Text("时间段一"),
                            #     dp.Text("时间段二")
                            # ],label="回测方案一"),
                        ],
                        type=dp.SelectType.DROPDOWN,
                )])

page_4 = dp.Page(title="标的池与选择工具",
                 blocks = [
                     dp.Text("标的池与选择工具")
                 ])

page_5 = dp.Page(title="风险敞口与因子工具",
                 blocks = [
                     dp.Text("风险敞口与因子工具")
                 ])

view = dp.View(
    page_1,
    page_2,
    page_3,
    page_4,
    page_5
)

dp.save_report(view, 
                path = pathlib.Path("report.html"),
                open=False)
        



# %%
# portfolio.portfolio_df.to_csv("src/portfolio1.csv")
# df_indicator.to_csv("src/df_indicator1.csv")