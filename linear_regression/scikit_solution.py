import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

olympics = sio.loadmat('dataset/olympics.mat')
male200 = olympics['male200']
female200 = olympics['female200']


m_year, m_winner_time = male200[:,0],male200[:,1]
f_year, f_winner_time = female200[:,0],female200[:,1]


# fit linear regression model
model = LinearRegression()
model.fit(m_year.reshape(-1,1),m_winner_time)
print(model.coef_,model.intercept_)
x_pred = np.array([2012]).reshape(-1,1)
y_pred = model.predict(x_pred)
print(f'year: 2012 Winning time: {y_pred}')

# TODO: fit linear regression model for women's 200m 
# TODO: what year does the model predict gender parity? (when will women catch up with men?)

### visualize results
# plot training data
plt.scatter(m_year,m_winner_time,label='men')
plt.scatter(f_year,f_winner_time,label='women')
# plot prediction
plt.scatter(x_pred,y_pred)

x_train = m_year.reshape(-1,1)
y_pred = model.predict(x_train)

plt.plot(x_train, y_pred)

x_test = np.arange(2008,2024,4).reshape(-1,1)
y_pred = model.predict(x_test)

plt.plot(x_test,y_pred,linestyle='dotted')

plt.xlabel('Olympic Year')
plt.ylabel('Running 200m Winning Time (in secs) ')
plt.legend()
plt.show()