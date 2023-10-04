import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np

df=pd.read_csv('Walmart.csv')

def moving_weighted_average(data, window_size):
    weights = np.arange(1, window_size + 1)
    smoothed = np.convolve(data, weights, 'valid') / weights.sum()
    return smoothed

window_size = 10  # Adjust the window size as needed
smoothed_data = moving_weighted_average(df['Weekly_Sales'], window_size)

plt.figure(figsize=(10, 6))
plt.plot(df['Weekly_Sales'], label='Original Data', color='blue')
plt.plot(smoothed_data, label='Smoothed Data', color='red')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.title('Original vs. Smoothed Data')
plt.legend()
plt.grid(True)
plt.show()









