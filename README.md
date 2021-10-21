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


<img src="https://user-images.githubusercontent.com/91713489/138058810-b58c406e-9c81-48f7-9fea-9a08e671cf5a.jpg" width="700" height="370">


Variation Auto Encoder is divided into encoder and decoder. The encoder makes z to extract the features of the data, and the decoder makes the model to make the original data using z. For more information, find to the reference.

ref: Pu Y, Gan Z, Henao R, Yuan X, Li C, Stevens A, et al. Variational autoencoder for deep learning of images, labels and captions. 

#### forecasting model

If you click the folder, there is another folder that divided into Jeju island, land and solar and wind. When you enter the land_solar folder, there are a result folder that stores results, a save_model folder that stores weight of the machine learning model, a machine learning model script, and raw data used for learning. Other folders also have the same structure.

The machine learning model script includes LSTM, GRU, DNN, and ARIMA. A brief description of each model is as follows.

Long Short-Term Memory (LSTM),a type of artificial circulation neural network (RNN), has the following structure:

<img src="https://user-images.githubusercontent.com/91713489/138211197-90b88166-9e2f-40cd-b238-849a01375627.jpg" width="700" height="370">

LSTM is a model that adress the problem of vanishing gradient of RNN and consists of cell, input gate, output gate, and forget gate. The cell controls the flow of information in the following three gates. The LSTM model is suitable for classifying, processing, and predicting time series data because there may be unknown delays between time series data. For more information, see script comment or refrence.

ref: Abbasimehr H, Shabani M, Yousefi M. An optimized model using LSTM network for demand forecasting.

The Gated Recurrent Unit(GRU) network is similar to the LSTM network but it has only two gates, each of which is the reset gate and the update gate. 

<img src="https://user-images.githubusercontent.com/91713489/138211272-f71cf03f-8644-4a60-9cba-9cc588e7c1a9.jpg" width="700" height="370">

The Deep Neural Network(DNN) that is illustrated below fig in a simple manner contains multiple hidden layers in-between the input and output layers. It is able to extract a higher level abstraction of the input data (i.e., a feature map) to preserve significant information distributed in each layer. In doing so, the DNN network learns arbitrary complex mappings between inputs and outputs to implement time-series forecasting.

ref: Lim JY, Safder U, How BS, Ifaei P, Yoo CK. Nationwide sustainable renewable energy and Power-to-X deployment planning in South Korea assisted with forecasting model. 

<img src="https://user-images.githubusercontent.com/91713489/138211280-3baab93c-baeb-44a7-a2a5-f59f50597509.jpg" width="700" height="370">

ref: Lim JY, Safder U, How BS, Ifaei P, Yoo CK. Nationwide sustainable renewable energy and Power-to-X deployment planning in South Korea assisted with forecasting model. 

ARIMA

ref: Li G, Hari SKS, Sullivan M, Tsai T, Pattabiraman K, Emer J, et al. Understanding error propagation in Deep Learning Neural Network (DNN) accelerators and applications. 

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
