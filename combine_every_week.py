import pandas as pd
from datetime import datetime
import numpy as np
#合并第一周和第二周的基金数据
week1=pd.read_csv('./data/fundProduct.csv')
week2=pd.read_csv('./data/week2.csv')
week2.drop('Unnamed: 0',axis=1,inplace=True)
week2.rename(columns={'fund_id':'Fund_ID'},inplace=True)

# print(week1.columns.values)
# print(week2.columns.values)
week=pd.concat([week1,week2],axis=0,ignore_index=True)#连接两个df,ignore_index=True表示不保留连接轴上的索引，产生一组新索引range(total_length)
week['Date']=pd.to_datetime(week['Date'],format='%Y/%m/%d')#把str变为time,才能比较
week.drop_duplicates(inplace=True)  #删除重复行
week.sort_values(by=['Fund_ID','Date'],inplace=True)    #先按Fund_ID排序，再按Date排序
week.reset_index(drop=True,inplace=True)    #把索引再重新排一下

print(week.head())
week.to_csv('./data/week1_2.csv')