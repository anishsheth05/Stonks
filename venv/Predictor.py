import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import pickle as pkl
import random as rd
import sklearn as sk
import sklearn.linear_model as lm
register_matplotlib_converters()

# above are all import statements, below are the functions


def train_model(model, X_data, Y_data, times=10, data_length_range=np.array([50, 300])):
    data_length = rd.randint(data_length_range[0], data_length_range[1])
    x_data = X_data.tail(data_length + negative_shift).head(data_length)
    y_data = Y_data.tail(data_length + negative_shift).head(data_length)
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x_data, y_data)
    best_model = model.fit(x_train, y_train)
    best = model.score(x_test, y_test)
    for i in range(times):
        data_length = rd.randint(data_length_range[0], data_length_range[1])
        x_data = X_data.tail(data_length + negative_shift).head(data_length)
        y_data = Y_data.tail(data_length + negative_shift).head(data_length)
        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x_data, y_data)
        model = model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        if acc > best:
            acc = best
            best_model = model
    print(best)
    return best_model


# all the variables
indicators = pd.read_excel('graph_data.xlsx')
indicators.index.key = 'Date'
days_ahead = 1
predict_this = 'Close'
acc_want = 0.987
negative_shift = 2
test_length = 150
X = indicators[['RSI', '20-Day Moving Average', '%B',
                'Top Bollinger Band', 'Bottom Bollinger Band', 'Standard Deviation', 'Trading Position']]
Y = indicators['Close'].shift(-negative_shift)
x = X.tail(test_length + negative_shift).head(test_length)
y = Y.tail(test_length + negative_shift).head(test_length)
pickled = open('stock_predictor.pickle', 'rb')

if pkl.load(pickled).score(x, y) > acc_want:
    try:
        predictor = pkl.load(pickled)
    except EOFError:
        predictor = lm.RidgeCV(cv=3)
        predictor = train_model(predictor, X, Y, 100)
        with open('stock_predictor.pickle', 'wb') as file:
            pkl.dump(predictor, file)
else:
    predictor = lm.RidgeCV(cv=3)
    predictor = train_model(predictor, X, Y, 100)
    with open('stock_predictor.pickle', 'wb') as file:
        pkl.dump(predictor, file)
print(predictor.score(x, y), predictor.coef_, predictor.intercept_)

plt.plot(indicators['Date'].tail(450), predictor.predict(X.tail(450)), label='Ridge Regression')
plt.plot(indicators['Date'].tail(450), indicators['Close'].tail(450), label='Actual Value')
plt.legend()
plt.show()

'''pickle_in = open('stock_predictor.pickle', 'rb')
ln = pkl.load(pickle_in)
if days_ahead > 100:
    pass
datas = indicators.drop([predict_this], 1)
indicators = indicators.tail(450).head(449)
X = indicators.drop([predict_this], 1)
y = indicators[predict_this]
accuracy1 = ln.score(X, y)
best = 0
while accuracy1 < acc_want:
    ln = lm.LinearRegression()
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.1)
    ln.fit(x_train, y_train)
    accuracy_score = ln.score(X, y)
    if accuracy_score > best:
        best = accuracy_score
        accuracy1 = ln.score(X, y)
        with open('stock_predictor.pickle', 'wb') as writer:
            pkl.dump(ln, writer)
ln = pkl.load(open('stock_predictor.pickle', 'rb'))
print(accuracy1)
print('Coefficient: \n', ln.coef_)
print('Intercept: \n', ln.intercept_)
future_index = pd.date_range(start=(dt.datetime(datas.tail(1).index.year[0], 1, 1) +
                                    dt.timedelta(int(str(datas.tail(1).index.day[0])))),
                             end=(dt.datetime(datas.tail(1).index.year[0] + 1, 1, 1) +
                                  dt.timedelta(int(str(datas.tail(1).index.day[0])))), freq='B')
for f in range(days_ahead):
    future = pd.DataFrame(ln.predict(datas.tail(1)),
                          index=future_index[f:f + 1]).rename(columns={0: 'Yesterday\'s Close'})
    datas.append(other=future['Yesterday\'s Close'])
    print(future_index[f])
    print(future)
print(indicators['Yesterday\'s Close'].tail(5), indicators['Close'].tail(5))
pd.DataFrame(data=ln.predict(X), index=X.index).rename(columns={0: 'Predictions'}).shift(1).append(future.rename(columns={'Yesterday\'s Close': 'Predictions'}))'''
