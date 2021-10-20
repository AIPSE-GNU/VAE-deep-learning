# 제목
***

Script for Yoonjae Lee et al. 2021, Journal, article name.
This repository contains a script for the sampling model and deep learning model.
In order to reproduce the results or figures, needs to download the all script from the git.

***
### git structure
#### sample generation model

Data conversion was used the conversion factor with reference to the Korean variable renewable energy policy(3020 plan). Converted data can be found in the VAE_DATA folder. Put it into VAE_MODEL/RES/train?RES_YB path, learn the model within ex_VAE_002_train.py, and generate sampling data using ex_VAE_002_test.py.
The structure of the VAE can be seen in the figure below:
![Fig 2  model structure_VAE](https://user-images.githubusercontent.com/91713489/138058810-b58c406e-9c81-48f7-9fea-9a08e671cf5a.jpg)

Variation Auto Encoder is divided into encoder and decoder. The encoder makes z to extract the features of the data, and the decoder makes the model to make the original data using z. For more information, find to the reference.

ref: Pu Y, Gan Z, Henao R, Yuan X, Li C, Stevens A, et al. Variational autoencoder for deep learning of images, labels and captions. 

#### forecasting model

If you click the folder, there is another folder that divided into Jeju island, land and solar and wind.
***
### Authors
Yoonjae Lee

Byeongmin Ha

Soonho Hwangbo

Department of Chemical Engineering, Gyeongsang National University

Home page: https://aipse.netlify.app

If you have any questions, please contact us.

***
copyright
