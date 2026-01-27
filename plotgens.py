import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


# Create a DataFrame
df = pd.read_csv('rewards.csv', header=None, names=['X', 'Y'])

# Select only the first x rows of data
#x = 43  # Replace 100 with the number of rows you want to select
#df = df.head(x)

# Calculate moving average with a window of 5
df['Moving_Average_5'] = df['Y'].rolling(window=5).mean()

# Calculate moving average with a window of 10
df['Moving_Average_10'] = df['Y'].rolling(window=10).mean()

# Calculate moving average with a window of 20
df['Moving_Average_20'] = df['Y'].rolling(window=20).mean()

# Calculate line of best fit
slope, intercept, r_value, p_value, std_err = linregress(df['X'], df['Y'])
df['Line_of_Best_Fit'] = slope * df['X'] + intercept
print('slope is: ', slope)

# Perform some additional analysis
# Calculate mean and standard deviation
mean_y = np.mean(df['Y'])
std_dev_y = np.std(df['Y'])

# Plotting
plt.figure(figsize=(12, 6))

# Original data points
#plt.scatter(df['X'], df['Y'], label='Original Data', color='blue')

# Moving average window 5
#plt.plot(df['X'], df['Moving_Average_5'], label='Moving Average (5)', color='orange')

# Moving average window 10
plt.plot(df['X'], df['Moving_Average_10'], label='Moving Average (10)', color='cyan')

# Moving average window 20
plt.plot(df['X'], df['Moving_Average_20'], 'green', label='Moving Average (20)', linewidth=2)

# Line of best fit
plt.plot(df['X'], df['Line_of_Best_Fit'], label='Line of Best Fit', color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Analysis')
plt.legend()
#plt.show()

plt.savefig('rewardsAnalyzeInprog.png')  # Specify your desired save path here
plt.close()  # Close the plot to free up memory

(slope, intercept, r_value, p_value, std_err), mean_y, std_dev_y