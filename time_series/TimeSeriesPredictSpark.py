#!/usr/bin/env python
# coding=utf-8
# @author 18099099
import datetime

import redis
from pyspark.sql import SparkSession

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA

import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import sys
import os
import logging
from logging import handlers

def get_logger(name):

    '''
    log日志输出格式方法
    :param name:
    :return:
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        pass
    else:
        formatter = logging.Formatter('[%(asctime)s][%(name)s] - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # backupCount: 保留最近5个
        filename = '{0}.log'.format("ts")
        th = handlers.RotatingFileHandler(filename=filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        th.setLevel(logging.DEBUG)
        th.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(th)
    return logger


def predict(x):

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
            # print(adfuller(ts, autolag='AIC'))
            return 0
        else:
            # series not stationarity
            # print(adfuller(ts, autolag='AIC'))
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

    print('\n')
    try:
        time_series_dict = dict(x.asDict().get("data"))
        # logger.debug(dict)
        df_org = pd.DataFrame(list(time_series_dict.values()), columns=['count'], index=pd.DatetimeIndex(time_series_dict.keys())).sort_index()
        start = 147
        end = start + 60
        df = df_org[start:end]
        # print(df.head())
        df_train = pd.Series(df['count'].values.astype(np.float), index=df.index)
    except Exception as e:
        print("get time series failed ! error msg {%s}"%(e))
        sys.exit(1)

    # 平稳性检查
    try:
        if_stationarity = stationarity_check(df_train)
        if if_stationarity == 1:
            diffts = diff_ts(df_train)
            diff1, train = diffts[0], diffts[1]
        else:
            train = df_train
    except Exception as e:
        print("stationarity check failed ! error msg {%s}"%(e))
    # print(train.head())

    # 模型训练
    try:
        model_val = proper_model(train)
        aic, p, q, model = model_val[0], model_val[1], model_val[2], model_val[3]
    except Exception as e:
        print("model fit failed ! error msg {%s}"%(e))
    # print(aic, p, q, model)

    # 模型预测
    try:
        future_interval = 4
        start_time = pd.date_range(start=train.index[-1], periods=2, freq='1Min')[-1]
        end_time = pd.date_range(start=start_time, periods=2, freq='%sMin'%(future_interval))[-1]
        predict_rst = model.predict(start_time, end_time, dynamic=True)
    except Exception as e:
        print("predict failed ! error msg {%s}"%(e))
    # print(predict_rst)

    # 预测结果还原
    try:
        if if_stationarity == 1:
            predict1 = pd.Series(diff1[-1:], index=diff1[-1:].index).append(predict_rst).cumsum()
            predict = pd.Series(df_train[-1:], index=df_train[-1:].index).append(predict1).cumsum()
            predict = predict.ix[predict_rst.index].astype(np.int)
        else:
            predict = predict_rst
    except Exception as e:
        print("predict val restore failed ! error msg {%s}"%(e))
    # print(predict)
    # # 计算精准率
    try:
        predt_df = pd.DataFrame(predict.values.astype(np.int), predict.index, columns=['predict'])
        predt_df['orignal'] = df_org.ix[predict.index]['count'].astype(np.int)
        predt_df['%ratio'] = np.fabs(predt_df['orignal']-predt_df['predict'])/predt_df['orignal']
        print(predt_df)
    except Exception as e:
        print("calc precision failed ! error msg {%s}"%(e))


if __name__ == '__main__':
    # logger = get_logger("TimeSeriesPredict")
    starttime = datetime.datetime.now()
    # 读取redis数据
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)

    # 将不同变量的序列以字典形式，插入list
    datas = []
    for i in range(10):
        # print "data"+(str(i)), "dts"+(str(i))
        data = "data"+(str(i+1))
        dts = "dts"+(str(i+1))
        time_series = "time_series_201809" + str(i+1).zfill(2)
        # print data, dts, time_series

        data = pd.read_msgpack(r.get(time_series))
        # print(data.head())
        #print data.shape, data[data.isnull().values==True]
        dts = pd.DataFrame(data['count'].values.astype(np.str), index=pd.to_datetime(data['time'].values).astype(np.str)).to_dict(orient='dict')
        new_dict = dict({"data":dts.get(0)})
        datas.append(new_dict)


    # print(datas)
    ## 将list进行map操作，将不同变量的序列分发到不同exector，进行spark预测
    spark = SparkSession.builder.appName("time_series_ml").\
        config("spark.executor.instances", 10).\
        config("spark.executor.memory", "1G").\
        master("local[11]").\
        getOrCreate()
    spark.createDataFrame(datas).rdd.map(predict).collect()

    # 运行总时间计算
    endtime = datetime.datetime.now()
    delta  = endtime - starttime
    print('long running time: %s s'%(delta.total_seconds()))

