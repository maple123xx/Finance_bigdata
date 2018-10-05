#-*-coding=utf-8-*-
import pandas as pd
import lightgbm as lgb
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

#加5条数据用于预测
def add_day(str):
    time = datetime.strptime(str, '%Y/%m/%d')
    time = time + timedelta(1)
    if time.weekday() > 4:
        time = time + timedelta(1)
    if time.weekday() > 4:
        time = time + timedelta(1)
    str = datetime.strftime(time, '%Y/%m/%d')
    return str
def X_add(X):
    time_now = datetime.strptime(X.index[-1], '%Y/%m/%d')#str to time
    time_now=time_now+timedelta(7)#加7天，下一次开盘是10月8日
    time_now=datetime.strftime(time_now,'%Y/%m/%d')#time to str
    for _ in range(5):
        time_now = add_day(time_now)
        X.loc[time_now] = X[-5:].apply(lambda x: x.mean())
    return X

def main():
    fund = fund_catagory()
    print(len(fund))
    y_pred=[]#222个基金未来5天的预测收盘价
    for i in range(len(fund)):
        fund[i]=fund[i][['close','BBI','MA','MACD','KDJ']]#特征选择
        X=fund[i][['BBI','MA','MACD','KDJ']]
        Y=fund[i]['close']
        X=X_add(X)
        X.fillna(method='ffill',inplace=True)
        X_scaled = preprocessing.scale(X)
        X_train = X_scaled[:-5]
        X_predict = X_scaled[-5:]
        y_train = Y

        print('Start training...')
        gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=1, n_estimators=25)
        gbm.fit(X_train, y_train, eval_metric='l1')
        print('Start predicting...')
        y_predict = gbm.predict(X_predict).tolist()
        y_pred.append(y_predict)

    return y_pred

if __name__=='__main__':
    y_pred=main()
    df_pred=pd.DataFrame(y_pred,columns=['day1','day2','day3','day4','day5'])
    df_pred.to_csv(r'./data/v1_predict.csv')


