import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense


df=pd.read_csv('Walmart.csv')

def moving_weighted_average(data, window_size):
    weights = np.arange(window_size, 0, -1)
    data_len = len(data)
    weight_len = len(weights)
    result_len = data_len + weight_len - 1
    result = np.zeros(result_len)

    for i in range(result_len):
        for j in range(weight_len):
            if i - j >= 0 and i - j < data_len:
                result[i] += data[i - j] * weights[j]
    smoothed = result / weights.sum()
    return smoothed

window_size = 10  # Adjust the window size as needed
smoothed_data = moving_weighted_average(df['Weekly_Sales'], window_size)


# Convert the NumPy array to a pandas DataFrame
smoothed_df = pd.DataFrame({'Smoothed_Weekly_Sales': smoothed_data})


diff=len(smoothed_df)-len(df)

smoothed_df = smoothed_df.iloc[:-diff]
smoothed_df['Smoothed_Weekly_Sales'] = smoothed_df['Smoothed_Weekly_Sales'].astype(str).str[:len(df)]


# Save the smoothed data to a CSV file
smoothed_df.to_csv('new.csv', index=False)  


# Read the first CSV file into a DataFrame (target file)
target_df = pd.read_csv('Walmart.csv')
target_df.to_csv('preprocessed_data.csv', index=False) 




# Specify the name of the column you want to replace
column_to_replace = 'Weekly_Sales'  # Replace with the name of the column

# Replace the column in the target DataFrame with the column from the source DataFrame
target_df['Weekly_Sales'] = smoothed_df['Smoothed_Weekly_Sales']

target_df.to_csv('preprocessed_data.csv', index=False) 

# Read the data from the CSV file with multiple columns
dataframe = pd.read_csv('preprocessed_data.csv', usecols=[2, 3, 4, 5, 6, 7], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Define the look-back window for multivariate time series
look_back = 1  # You can adjust this for a different sequence length
num_features = dataset.shape[1]

# Create function to prepare multivariate data
def create_multivariate_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# Prepare the data
trainX, trainY = create_multivariate_dataset(train, look_back)
testX, testY = create_multivariate_dataset(test, look_back)

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, num_features)))  # You can adjust the number of LSTM units
model.add(Dense(num_features))  # Output layer matches the number of features
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))



# Calculate R2 score for the training data
trainR2 = r2_score(trainY, trainPredict)
print('Training R2 Score: %.2f' % trainR2)

# Calculate R2 score for the test data
testR2 = r2_score(testY, testPredict)
print('Test R2 Score: %.2f' % testR2)


# Plot the results
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2):len(dataset) , :] = testPredict

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
