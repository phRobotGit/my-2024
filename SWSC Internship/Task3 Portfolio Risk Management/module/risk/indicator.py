import pandas as pd 
import numpy as np
from typing import Iterator

def calculate_max_drawdown(prices):
    # 这个计算可能有问题
    max_drawdown = 0
    peak = prices[0]  # 当前峰值
    trough = prices[0]  # 当前谷底

    for price in prices:
        if price > peak:
            peak = price
            trough = price
        elif price < trough:
            trough = price

        drawdown = (peak - trough) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

def calculate_indicator_price(price:Iterator) -> pd.Series:
    price = pd.Series(price)
    ret = np.log(price / price.shift())
    return(pd.Series({
        "mean return": ret.mean(),
        "90% VaR return": ret.quantile(0.10),
        "95% VaR return": ret.quantile(0.05),
        "volatility return": ret.std(),
        "max_drawdown (close price)": calculate_max_drawdown(price.tolist()),
    }))