import pandas as pd
import numpy as np
import random

df = pd.read_csv("./RES/vae_generative_test_jeju_wind_01.csv", header=None)
# print(df)

ind = np.arange(365*50)
random.seed(42)
random.shuffle(ind)

df_100 = np.empty((1, 24), object)
df_100 = pd.DataFrame(df_100)
# df_to = np.empty((1,24), object)
# df_to = pd.DataFrame(df_to)
k = 0
for i in range(365):
    for j in range(50):
        globals()['df{}'.format(i)] = df.iloc[k, :]
        df_100 = df_100.append(globals()['df{}'.format(i)], ignore_index=True)
        k += 1
        print(k)
    df_100 = df_100[1:]
    if i == 0:
        df_to = df_100
        df_100 = np.empty((1, 24), object)
        df_100 = pd.DataFrame(df_100)
    else:
        df_to = pd.concat([df_to, df_100], axis=1)
        df_100 = np.empty((1, 24), object)
        df_100 = pd.DataFrame(df_100)

print(df_to)
df_to.to_csv("sampling/jeju_wind_rs42/sampling_jeju_wind_50.csv", header=None, index=False)