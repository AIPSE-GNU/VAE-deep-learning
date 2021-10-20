# 제목

Script for Yoonjae Lee et al. 2021, Journal, article name.
This repository contains a script for the sampling model and deep learning model.
In order to reproduce the results or figures, needs to download the all script from the git.

### git structure
1. sample generation model
Data conversion was used the conversion factor with reference to the Korean variable renewable energy policy(3020 plan). Converted data can be found in the VAE_DATA folder. Put it into VAE_MODEL/RES/train?RES_YB path, learn the model within ex_VAE_002_train.py, and generate sampling data using ex_VAE_002_test.py.
VAE의 구조는 아래 그림과 같으며 자세한 내용은 reference에서 확인할 수 있다.
ref: Pu Y, Gan Z, Henao R, Yuan X, Li C, Stevens A, et al. Variational autoencoder for deep learning of images, labels and captions. 

2. forecasting model
If you click the folder, there is another folder that divided into Jeju island, land and solar and wind.

Authors
Yoonjae Lee
Byeongmin Ha
Soonho Hwangbo
Department of Chemical Engineering, Gyeongsang National University
https://aipse.netlify.app

If you have any questions, please contact us.


copyright
