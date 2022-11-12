import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numpy import ndarray

np.set_printoptions(suppress=True,precision=2)

olympics = sio.loadmat('dataset/olympics.mat')
male200 = olympics['male200']
female200 = olympics['female200']


m_year, m_winner_time = male200[:,0],male200[:,1]
f_year, f_winner_time = female200[:,0],female200[:,1]


# fit a linear regression model
class LinearRegression:
    def __init__(self) -> None:
        self.coef_:ndarray
        self.intercept_:float

    def fit(self,X:ndarray,y:ndarray):
        """_summary_

        Args:
            X (ndarray): 2D array, each training sample takes 1 row
            y (ndarray): 2D array, each y_i takes 1 row
        """
        rows,cols = X.shape
        X = np.c_[np.ones(rows),X]
        self.coef_ = np.linalg.inv(X.T@X)@ X.T@y
        self.intercept_ = self.coef_[0,0]
        self.coef_ = self.coef_[1]
    
    def predict(self,X: ndarray)->ndarray:
        return X@self.coef_ + self.intercept_

# fit linear regression model
model = LinearRegression()
X_train = np.array(m_year).reshape(-1,1)
y_train = np.array(m_winner_time).reshape(-1,1)
model.fit(X_train,y_train)
print(model.coef_,model.intercept_)
x_pred = np.array([2012]).reshape(-1,1)
y_pred = model.predict(x_pred)
print(f'year: 2012 Winning time: {y_pred}')

### visualize results
# plot training data
plt.scatter(m_year,m_winner_time,label='men')
plt.scatter(f_year,f_winner_time,label='women')
# plot prediction
plt.scatter(x_pred,y_pred)


y_pred = model.predict(X_train)
plt.plot(X_train, y_pred)

x_test = np.arange(2008,2024,4).reshape(-1,1)
y_pred = model.predict(x_test)

plt.plot(x_test,y_pred,linestyle='dotted')

plt.xlabel('Olympic Year')
plt.ylabel('Running 200m Winning Time (in secs) ')
plt.legend()
plt.show()