"""
Script: nn.regress

A simple example of linear regression in pure python completely from scratch.
"""

# Pizza sales dataset
income = [5, 10, 20, 8, 4, 6, 12, 15]
sales = [27, 46, 73, 40, 30, 28, 46, 59]

# Number of observations
n = len(income)

# Calculating averages
x_avg = sum(income) / n
y_avg = sum(sales) / n

# Calculating x^2 and x * y
x_squared = [x**2 for x in income]
xy = [x * y for x, y in zip(income, sales)]

x_squared_avg = sum(x_squared) / n
xy_avg = sum(xy) / n

# Calculating slope (m) and intercept (b) based on the formulas
m_hat = (xy_avg - x_avg * y_avg) / (x_squared_avg - x_avg**2)
b_hat = y_avg - m_hat * x_avg

print(m_hat, b_hat)

# Compute predicted pizza sales for each income
predicted_sales = [m_hat * x + b_hat for x in income]

print(predicted_sales)
