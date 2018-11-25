#-*-coding=utf-8-*-
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import xlrd
from datetime import datetime,timedelta

def fund_catagory():#将基金按照ID分开成222个基金
    fund_product = pd.read_csv(r'./data/week_pre8.csv', index_col='Date')
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
        fund[i]=fund[i][['close','NAV_adj','BBI','MA','MACD','KDJ']]#特征选择
        X=fund[i][['NAV_adj','BBI','MA','MACD','KDJ']]
        Y=fund[i]['close'].shift(-1)
        X.fillna(method='ffill',inplace=True)
        X=np.array(X)
        X.astype(np.float32)
        X_scaled = preprocessing.scale(X)
        X_train = X_scaled[:-1]
        X_predict = X_scaled[-1]
        y_train = np.array(Y.iloc[:-1])
        y_train.astype(np.float32)

        print('Start training {} fund'.format(i+1))
        # clf=LinearRegression(n_jobs=-1)
        # clf.fit(X_train,y_train)
        # y_predict = clf.predict(X_predict).tolist()
        # gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=1, n_estimators=25)
        # gbm.fit(X_train, y_train, eval_metric='l1')
        # y_predict = gbm.predict([X_predict]).tolist()
        # y_pred.append(y_predict)

        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
        model.fit(X_train, y_train)
        y_predict = model.predict([X_predict]).tolist()
        y_pred.append(y_predict)

    return y_pred

if __name__=='__main__':
    y_pred=main()
    df_pred = pd.DataFrame(y_pred, columns=['day1'])#预测的未来1天的股价
    col_name = df_pred.columns.tolist()
    col_name.insert(0,'day_old')    #每个股票最后一天的股价,放在第一列
    df_pred=df_pred.reindex(columns=col_name)

    fund = fund_catagory()
    for i in range(len(fund)):
        df_pred.ix[i,'day_old']=fund[i].ix[-1,'close']
    df_pred['diff']=df_pred['day1']-df_pred['day_old']
    #df_pred['stock_return'] = df_pred['diff'] / df_pred['day_old']
    df_pred.sort_values(by='diff',ascending=False,inplace=True)

    df_pred.to_csv(r'./data/Ngt5_lgb_week8.csv')


