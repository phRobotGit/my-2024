import pathlib
from typing import Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, AutoDateLocator

def plot_price_matrix(df_price_matrix:pd.DataFrame,
                    width_y_per:float = 3,  
                    width_x:float = 20,
                    save_path:Optional[Union[str, pathlib.Path]] = None,
                    ):
    df_plot = df_price_matrix
    
    ncols = 3 
    nrows = int(np.ceil(len(df_plot.columns)/ncols))
    
    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols, 
                             figsize=(nrows*width_y_per, width_x),)
    gen = (i for i in axes.reshape(-1) )

    for col in df_plot.columns:
        ax=next(gen)
        sns.lineplot(data=df_plot[col],
                    label="close", 
                    ax=ax)
        # 设置X轴刻度和标签格式
        date_locator = AutoDateLocator()
        date_formatter = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.xaxis.set_tick_params(rotation=45)
    
    if save_path != None:
        plt.savefig(save_path)
    
    return(fig)


def plot_combination_chart(df):
    # 设置画布和子图
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

    # 绘制AUM折线图
    df["aum"] = df["aum"] / df["aum"].values[0] * 100
    ax1.plot(df.index, df['aum'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('AUM')
    ax1.set_title('AUM Line')

    # 设置X轴刻度和标签格式
    date_locator = AutoDateLocator()
    date_formatter = DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_locator(date_locator)
    ax1.xaxis.set_major_formatter(date_formatter)
    ax1.xaxis.set_tick_params(rotation=45)

    # 绘制收益率分布直方图
    sns.histplot(df['return'], ax=ax2, kde=True)
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Return Distribution')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    # plt.show()
    return(fig)