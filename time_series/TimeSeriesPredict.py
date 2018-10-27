#!/usr/bin/env python
# coding=utf-8
# @author 18099099
import datetime

import redis

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA

import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import sys
import logging
logger = logging.getLogger("TimeSeriesPredict")


def stationarity_check(ts):

    """
    check time series whether or not stationarity
    :param ts: time series data
    :return:
    """
    adfuller_val = adfuller(ts, autolag='AIC')[0:5]
    test_statistic = adfuller_val[0]
    p_value = adfuller_val[1]
    vritical_val1 = adfuller_val[4]['1%']
    vritical_val2 = adfuller_val[4]['5%']
    vritical_val3 = adfuller_val[4]['10%']

    if (test_statistic < vritical_val1 and test_statistic < vritical_val2 and test_statistic < vritical_val3) and (np.fabs(p_value - 0) <= 0.01):
        # series is stationarity
        print(adfuller(ts, autolag='AIC'))
        return 0
    else:
        # series not stationarity
        print(adfuller(ts, autolag='AIC'))
        return 1


def diff_ts(ts, interval=1):

    """
    diff time series for stationarity
    :param ts:
    :param interval:
    :return:
    """
    diff1 = ts.diff(1)
    diff1.dropna(inplace=True)
    diff2 = diff1.diff(1)
    diff2.dropna(inplace=True)
    return diff1, diff2


def proper_model(data_ts, maxLag=10):

    """
    init ARMA model
    :param data_ts:
    :param maxLag:
    :return:
    """

    import warnings
    warnings.filterwarnings('ignore')

    init_bic = sys.maxsize
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel


if __name__ == '__main__':

    starttime = datetime.datetime.now()

    # redis读取最近60s数据
    r = redis.Redis(host = '127.0.0.1', port = 6379, db = 0)
    data = pd.read_msgpack(r.get("time_series"))
    dfOrg = pd.DataFrame(data['count'].values, columns=['count'], index=data.time)
    # print dfOrg
    df = dfOrg[1230:1290]
    dfTrain = pd.Series(df['count'].values.astype(np.float), index=df.index)

    # 平稳性检查
    if_stationarity = stationarity_check(dfTrain)
    if if_stationarity == 1:
         diffts = diff_ts(dfTrain)
         diff1, train = diffts[0], diffts[1]
    else:
         train = dfTrain
    # print train

    # 模型训练
    model_val = proper_model(train)
    aic, p, q, model = model_val[0], model_val[1], model_val[2], model_val[3]
    print(aic, p, q, model)

    # 模型预测
    future_interval = 4
    start_time = pd.date_range(start=train.index[-1], periods=2, freq='1Min')[-1]
    end_time = pd.date_range(start=start_time, periods=2, freq='%sMin'%(future_interval))[-1]
    predict_rst = model.predict(start_time, end_time, dynamic=True)
    print(predict_rst)

    # 预测结果还原
    if if_stationarity == 1:
        predict1 = pd.Series(diff1[-1:], index=diff1[-1:].index).append(predict_rst).cumsum()
        predict = pd.Series(dfTrain[-1:], index=dfTrain[-1:].index).append(predict1).cumsum()
        predict = predict.ix[predict_rst.index].astype(np.int)
    else:
        predict = predict_rst
    print(predict)
    # 计算精准率
    predt_df = pd.DataFrame(predict.values.astype(np.int), predict.index, columns=['predict'])
    predt_df['orignal'] = dfOrg.ix[predict.index]['count']
    predt_df['%ratio'] = np.fabs(predt_df['orignal']-predt_df['predict'])/predt_df['orignal']
    print(predt_df)

    # 运行总时间计算
    endtime = datetime.datetime.now()
    delta  = endtime - starttime
    print('long running time: %s s'%(delta.total_seconds()))