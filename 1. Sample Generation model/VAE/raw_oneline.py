import pandas as pd
import numpy as np

js = pd.read_csv("./RES/train/RES_YB/raw_data_jeju_solar(2030).csv", index_col=0)
jw = pd.read_csv("./RES/train/RES_YB/raw_data_jeju_wind(2030).csv", index_col=0)
ls = pd.read_csv("./RES/train/RES_YB/raw_data_land_solar(2030).csv", index_col=0)
lw = pd.read_csv("./RES/train/RES_YB/raw_data_land_wind(2030).csv", index_col=0)
# print(js)
js_ol = np.empty((1, 1), object)
# js_ol = pd.DataFrame(js_ol)
jw_ol = np.empty((1, 1), object)
# jw_ol = pd.DataFrame(jw_ol)
ls_ol = np.empty((1, 1), object)
# ls_ol = pd.DataFrame(ls_ol)
lw_ol = np.empty((1, 1), object)
# lw_ol = pd.DataFrame(lw_ol)
li = [js, jw, ls, lw]

for j in range(365):
    js_li = js.iloc[j, :]
    for k in range(24):
        a = js_li.iloc[k]
        a = np.reshape(a, (1, 1))
        js_ol = np.append(js_ol, a, axis=0)
js_ol = js_ol[1:]
js_ol = pd.DataFrame(js_ol)

for j in range(365):
    js_li = jw.iloc[j, :]
    for k in range(24):
        a = js_li.iloc[k]
        a = np.reshape(a, (1, 1))
        jw_ol = np.append(jw_ol, a, axis=0)
jw_ol = jw_ol[1:]
jw_ol = pd.DataFrame(jw_ol)


for j in range(365):
    ls_li = ls.iloc[j, :]
    for k in range(24):
        a = ls_li.iloc[k]
        a = np.reshape(a, (1, 1))
        ls_ol = np.append(ls_ol, a, axis=0)
ls_ol = ls_ol[1:]
ls_ol = pd.DataFrame(ls_ol)

for j in range(365):
    lw_li = lw.iloc[j, :]
    for k in range(24):
        a = lw_li.iloc[k]
        a = np.reshape(a, (1, 1))
        lw_ol = np.append(lw_ol, a, axis=0)
lw_ol = lw_ol[1:]
lw_ol = pd.DataFrame(lw_ol)

total = pd.concat((js_ol, jw_ol, ls_ol, lw_ol), axis=1)
print(total)

total.to_csv("raw_total.csv")