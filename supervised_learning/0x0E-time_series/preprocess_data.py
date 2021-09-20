#!/usr/bin/env python3
"""preprocess raw csv data"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')



# Unix-time to
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
print(df.info())
# Resampling to daily frequency
df.index = df.Timestamp
# df = df.resample('H').mean()
# print(df['Weighted_price'].isnull())

# Resampling to hourly frequency
df_hour = df.resample('H').mean()
print('####')
null_values = df_hour['Weighted_Price'].isnull().sum()
print(df_hour['Weighted_Price'].notnull().sum())
length_data = len(df_hour['Weighted_Price'])
# if less than 5% off rows have null values drop them otherwise interpolate
if null_values / length_data <= 0.5:
    df_hour = df_hour.dropna()
else:
    df_hour = df_hour.interpolate()


print('####')
print('total rows {}'.format(len(df_hour['Weighted_Price'])))
print(df_hour.describe().transpose())
print(df_hour.head())

# Resampling to annual frequency
df_year = df.resample('A-DEC').mean()

# Resampling to quarterly frequency
df_Q = df.resample('Q-DEC').mean()

# PLOTS
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

plt.subplot(221)
plt.plot(df_hour.Weighted_Price, '-', label='By Hours')
plt.legend()

plt.subplot(222)
plt.plot(df_hour.Weighted_Price, '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(df_Q.Weighted_Price, '-', label='By Quarters')
plt.legend()

plt.subplot(224)
plt.plot(df_year.Weighted_Price, '-', label='By Years')
plt.legend()

# plt.tight_layout()
plt.show()
