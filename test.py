import pandas as pd
import numpy as np

# #将第二周的收益率排序
# week2=pd.read_csv('./data/week2.csv')
# week2.drop('Unnamed: 0',axis=1,inplace=True)
# fund_num=week2['fund_id'].groupby(week2.fund_id).count()
# fund = []
# for i in range(fund_num.shape[0]):
#     each_fund = week2[week2['fund_id'].isin([i + 1])]
#     fund.append(each_fund)
#
# print(fund[0].ix[fund[0]['Date']=='2018/9/28','close'].values)
# return_rate=[]
# for i in range(len(fund_num)):#222个基金
#     rate=(fund[i].ix[fund[i]['Date']=='2018/10/12','close'].values)/\
#          (fund[i].ix[fund[i]['Date']=='2018/9/28','close'].values)-1
#     rate=rate.tolist()
#     return_rate.append(rate)
# # return_rate=np.array(return_rate)
# # print(return_rate.shape)
# return_rate=[n for a in return_rate for n in a] #二维list转一维list
# print(return_rate)
# print('222个基金的平均收益率：{:.2%}'.format(np.array(return_rate).mean()))
# print('222个基金中收益率最高为{}号：{:.2%}'.format(71,np.array(return_rate).max()))
# print('222个基金中收益率最低为{}号：{:.2%}'.format(58,np.array(return_rate).min()))
# for i in [4,26,54,73,75,83,133,185]:
#     print('{}号基金的收益率为{:.2%}'.format(i+1,return_rate[i]))

week_pre5=pd.read_csv('./data/week_pre5.csv')#找出10月10号到10月11号上涨的基金
fund_num=week_pre5['Fund_ID'].groupby(week_pre5.Fund_ID).count()
fund = []
for i in range(fund_num.shape[0]):
    each_fund = week_pre5[week_pre5['Fund_ID'].isin([i + 1])]
    fund.append(each_fund)
print(fund[0].ix[fund[0]['Date']=='2018-10-10','close'].values)
up=[]
for i in range(len(fund)):
    if fund[i].ix[fund[i]['Date']=='2018-10-10','close'].values<fund[i].ix[fund[i]['Date']=='2018-10-11','close'].values:
        up.append(i+1)
print(up)
