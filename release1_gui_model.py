#-*-coding=utf-8-*-
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import xlrd
from datetime import datetime,timedelta

def fund_catagory():#将基金按照ID分开成222个基金
    fund_product = pd.read_csv(r'./data/fundProduct.csv', index_col='Date')
    financial_product = pd.read_excel(r'./data/financialProduct.xlsx')
    fund_num = fund_product['Fund_ID'].groupby(fund_product.Fund_ID).count()
    fund = []
    for i in range(fund_num.shape[0]):
        each_fund = fund_product[fund_product['Fund_ID'].isin([i + 1])]
        fund.append(each_fund)
    return fund

def main():
    fund = fund_catagory()
    y_pred=[]#222个基金未来5天的预测收盘价
    for i in range(len(fund)):
        fund[i]=fund[i][['close','BBI','MA','MACD','KDJ']]#特征选择
        X=fund[i][['BBI','MA','MACD','KDJ']]
        Y=fund[i]['close'].shift(-5)
        X.fillna(method='ffill',inplace=True)
        X=np.array(X)
        X_scaled = preprocessing.scale(X)
        X_train = X_scaled[:-5]
        X_predict = X_scaled[-5:]
        y_train = np.array(Y.iloc[:-5])

        print('Start training {} fund'.format(i+1))
        # clf=LinearRegression(n_jobs=-1)
        # clf.fit(X_train,y_train)
        # y_predict = clf.predict(X_predict).tolist()
        gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=1, n_estimators=25)
        gbm.fit(X_train, y_train, eval_metric='l1')
        y_predict = gbm.predict(X_predict).tolist()
        y_pred.append(y_predict)

    return y_pred

if __name__=='__main__':
    #hello world
    y_pred=main()
    fund = fund_catagory()
    for i in range(len(fund)):
        fund[i].join
    df_pred=pd.DataFrame(y_pred,columns=['day1','day2','day3','day4','day5'])
    df_pred.to_csv(r'./data/v1_gui_predict.csv')


