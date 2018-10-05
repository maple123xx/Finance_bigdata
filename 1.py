import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,shuffle=False, random_state=42)
print(X_train)
# start_test = datetime.datetime(2005,1,1)
# print(start_test)
# df=pd.DataFrame({'b':[1,4,6,5,8,10,23,24,34]})
# print(df)
# df['shift1']=df['b'].shift(5)
# df['diff']=df['b']-df['shift1']
# print(df)
#
# X=np.array(range(670))
# #X=X.reshape((670,1))
# Y=np.array(close1)
# #Y=Y.reshape(670,1)
# z=np.polyfit(X,Y,4)
# print(z)
# p = np.poly1d(z)
# print(p(671))
# xp=np.linspace(0,670)
#
# fig=plt.figure()
# plt.plot(p(xp))
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=1)
# reg=linear_model.LinearRegression()
# reg.fit(X_train,y_train)
# # y_pred = reg.predict(X_test)
# # print(y_pred)
# y_pred=reg.predict([[671]])
# print(y_pred)
