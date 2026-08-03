import pandas as pd

df = pd.read_csv(r"D:\Retail-Price-Optimization\retail.csv")

print(df.head())
X = df[['price','competitor_price','discount','cogs','month']]
y = df['sales']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred[:5])
import numpy as np

price_range = np.arange(50, 500, 5)

best_price = 0
max_profit = 0

for price in price_range:
    demand = model.predict([[price, 200, 10, 100, 6]])[0]
    profit = (price - 100) * demand

    if profit > max_profit:
        max_profit = profit
        best_price = price

print("Best Price:", best_price)
print("Max Profit:", max_profit)
import matplotlib.pyplot as plt

prices = []
profits = []

for price in range(50, 500, 10):
    demand = model.predict([[price, 200, 10, 100, 6]])[0]
    profit = (price - 100) * demand

    prices.append(price)
    profits.append(profit)

plt.plot(prices, profits)
plt.xlabel("Price")
plt.ylabel("Profit")
plt.title("Price vs Profit Curve")
plt.show()
