import tensorflow as tf
import keras
from tensorflow.keras.layers import Layer, Dot, Activation
from tensorflow.keras.layers import Input, Conv1D, Activation, SpatialDropout1D, Dense, Add, Multiply, Lambda, \
    GlobalAveragePooling1D
from keras.layers import Input, LSTM, Dropout, Dense, GRU, Conv1D, Activation, GlobalMaxPooling1D, MaxPooling1D, \
    Flatten, concatenate, Permute, Reshape, Multiply, Add, Bidirectional,Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Attention, add
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape


def easy_result(y_train, y_train_predict, train_index, model_index, col_index):
    plt.figure(figsize=(10, 5))
    plt.plot(y_train[:])
    plt.plot(y_train_predict[:])
    plt.legend(('real', 'predict'), fontsize='15')
    plt.title("%s Data" % train_index, fontsize='20')
    plt.show()
    print('\n')

    plot_begin, plot_end = min(min(y_train), min(y_train_predict)), max(max(y_train), max(y_train_predict))
    plot_x = np.linspace(plot_begin, plot_end, 10)
    plt.figure(figsize=(5, 5))
    plt.plot(plot_x, plot_x)
    plt.plot(y_train, y_train_predict, 'o')
    plt.title("%s Data" % train_index, fontsize='20')
    plt.show()

    # 输出结果
    print('%s上的MAE/RMSE/MAPE/R^2' % train_index)
    print(mean_absolute_error(y_train, y_train_predict))
    print(np.sqrt(mean_squared_error(y_train, y_train_predict)))
    print(mape(y_train, y_train_predict))
    print(r2_score(y_train, y_train_predict))

    pred_data = np.vstack([y_train, y_train_predict])
    pred_data = pd.DataFrame(pred_data).T
    return mean_absolute_error(y_train, y_train_predict), np.sqrt(mean_squared_error(y_train, y_train_predict))


data_1 = pd.read_excel('ship1-1.xlsx').iloc[:, 0:9]
data_2 = data_1.iloc[:, 1:]
data_3 = data_1.iloc[:, 0:1]
test_ratio = 0.2
windows =

mm_x = MinMaxScaler()
mm_y = MinMaxScaler()
data_2 = mm_x.fit_transform(data_2)
data_3 = mm_y.fit_transform(data_3)


# print(data_3.shape[0])
def convert_data(data_L, data_S):
    data_L = np.array(data_L)
    data_S = np.array(data_S)
    cut = round(test_ratio * data_L.shape[0])
    amount_of_features = data_L.shape[1]
    lstm_input = []
    lstm_output = []
    for i in range(len(data_L) - windows):
        lstm_input.append(data_L[i:i + windows, :])
        lstm_output.append(data_S[i + windows, :])
    lstm_input = np.array(lstm_input)
    lstm_output = np.array(lstm_output)
    x_train, y_train, x_test, y_test = \
        lstm_input[:-cut, :, :], lstm_output[:-cut:], lstm_input[-cut:, :, :], lstm_output[-cut:]
    print('x_train.shape', x_train.shape)
    print('x_test.shape', x_test.shape)
    print('y_train.shape', y_train.shape)
    print('y_test.shape', y_test.shape)
    return x_train, x_test, y_train, y_test, amount_of_features


x_train, x_test, y_train, y_test, amount_of_features = convert_data(data_2, data_3)

P_x_train = x_train[:,:,1:3]
P_x_test = x_test[:,:,1:3]
S_x_train = x_train[:,:,0:1]
S_x_test = x_test[:,:,0:1]

TRIM_x_train = x_train[:,:,3:4]
TRIM_x_test = x_test[:,:,3:4]
DRAFT1_x_train = x_train[:,:,4:5]
DRAFT1_x_test = x_test[:,:,4:5]
DRAFT2_x_train = x_train[:,:,5:6]
DRAFT2_x_test = x_test[:,:,5:6]
HW_x_train = x_train[:,:,6:7]
HW_x_test = x_test[:,:,6:7]
CW_x_train = x_train[:,:,7:8]
CW_x_test = x_test[:,:,7:8]
PTCN_x_train = x_train[:,:,3:8]
PTCN_x_test = x_test[:,:,3:8]

class Loc_Attention(tf.keras.layers.Layer):
    def __init__(self, dim_input, dim_q, dim_v):
        super(Loc_Attention, self).__init__()

        self.dim_input = dim_input
        self.dim_q = dim_q
        self.dim_k = dim_q
        self.dim_v = dim_v

        self.linear_q = Dense(self.dim_q, use_bias=False)
        self.linear_k = Dense(self.dim_k, use_bias=False)
        self.linear_v = Dense(self.dim_v, use_bias=False)
        self.norm_fact = 1 / tf.math.sqrt(float(self.dim_k))

    def call(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dist = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) * self.norm_fact
        dist = tf.nn.softmax(dist, axis=-1)
        att = tf.matmul(dist, v)
        att = att[:, 0, :]
        return att


class G_Attention(tf.keras.layers.Layer):
    def __init__(self, dim_input, dim_q, dim_v):
        super(G_Attention, self).__init__()

        self.dim_input = dim_input
        self.dim_q = dim_q
        self.dim_k = dim_q
        self.dim_v = dim_v

        self.linear_q = Dense(self.dim_q, use_bias=False)
        self.linear_k = Dense(self.dim_k, use_bias=False)
        self.linear_v = Dense(self.dim_v, use_bias=False)
        self.norm_fact = 1 / tf.math.sqrt(float(self.dim_k))

    def call(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dist = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) * self.norm_fact
        dist = tf.nn.softmax(dist, axis=-1)
        att = tf.matmul(dist, v)
        att = att
        return att


def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)
    o = Activation('relu')(r)  # 激活函数
    return o


def FResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    r = Conv2D(filters, kernel_size, padding='valid', dilation_rate=dilation_rate)(r)
    o = Activation('relu')(r)  # 激活函数
    return o

num_hidden1 =
num_hidden2 =
num_Loc_att =
num_g_att =
num_feature =
num_flitter =
windows =
num_batch =
num_epoch =

def get_model():
    input_layer_P = Input(shape=(windows,2))
    input_layer_H = Input(shape=(windows,1))
    o_P = LSTM(num_hidden1, return_sequences=True)(input_layer_P)
    o_H = LSTM(num_hidden1, return_sequences=True)(input_layer_H)
    SA_P =  Loc_Attention(num_hidden1,windows,num_Loc_att)
    SA_H =  Loc_Attention(num_hidden1,windows,num_Loc_att)

    o_P = SA_P(o_P)
    o_H = SA_H(o_H)

    input_layer_TRIM = Input(shape=(windows,1))
    input_layer_DRAFT1 = Input(shape=(windows,1))
    input_layer_DRAFT2 = Input(shape=(windows,1))
    input_layer_HW = Input(shape=(windows,1))
    input_layer_CW = Input(shape=(windows,1))
    input_layer_PTCN = Input(shape=(windows,5))

    SA_TRIM =  Loc_Attention(num_flitter,windows,num_Loc_att)
    SA_DRAFT1 =  Loc_Attention(num_flitter,windows,num_Loc_att)
    SA_DRAFT2 =  Loc_Attention(num_flitter,windows,num_Loc_att)
    SA_HW =  Loc_Attention(num_flitter,windows,num_Loc_att)
    SA_CW =  Loc_Attention(num_flitter,windows,num_Loc_att)
    SA_PTCN =  Loc_Attention(*,*,*)#Set attention mechanism parameters according to actual conditions

    x1_TRIM=ResBlock(input_layer_TRIM,filters=num_flitter,kernel_size=8,dilation_rate=1)
    x1_TRIM=ResBlock(x1_TRIM,filters=num_flitter,kernel_size=4,dilation_rate=2)
    x1_TRIM=ResBlock(x1_TRIM,filters=num_flitter,kernel_size=2,dilation_rate=4)
    o_TRIM = SA_TRIM(x1_TRIM)
    x1_DRAFT1=ResBlock(input_layer_DRAFT1,filters=num_flitter,kernel_size=8,dilation_rate=1)
    x1_DRAFT1=ResBlock(x1_DRAFT1,filters=num_flitter,kernel_size=4,dilation_rate=2)
    x1_DRAFT1=ResBlock(x1_DRAFT1,filters=num_flitter,kernel_size=2,dilation_rate=4)
    o_DRAFT1=SA_DRAFT1(x1_DRAFT1)
    x1_DRAFT2=ResBlock(input_layer_DRAFT2,filters=num_flitter,kernel_size=8,dilation_rate=1)
    x1_DRAFT2=ResBlock(x1_DRAFT2,filters=num_flitter,kernel_size=4,dilation_rate=2)
    x1_DRAFT2=ResBlock(x1_DRAFT2,filters=num_flitter,kernel_size=2,dilation_rate=4)
    o_DRAFT2=SA_DRAFT2(x1_DRAFT2)
    x1_HW=ResBlock(input_layer_HW,filters=num_flitter,kernel_size=8,dilation_rate=1)
    x1_HW=ResBlock(x1_HW,filters=num_flitter,kernel_size=4,dilation_rate=2)
    x1_HW=ResBlock(x1_HW,filters=num_flitter,kernel_size=2,dilation_rate=4)
    o_HW = SA_HW(x1_HW)
    x1_CW=ResBlock(input_layer_CW,filters=num_flitter,kernel_size=8,dilation_rate=1)
    x1_CW=ResBlock(x1_CW,filters=num_flitter,kernel_size=4,dilation_rate=2)
    x1_CW=ResBlock(x1_CW,filters=num_flitter,kernel_size=2,dilation_rate=4)
    o_CW=SA_CW(x1_CW)

    x1_PTCN = Reshape((num_hidden1, windows, 1))(input_layer_PTCN)
    x1_PTCN=FResBlock(x1_PTCN,filters=num_flitter,kernel_size=(4,3),dilation_rate=(1,1))
    x1_PTCN=FResBlock(x1_PTCN,filters=num_flitter,kernel_size=(2,3),dilation_rate=(1,2))
    x1_PTCN=FResBlock(x1_PTCN,filters=num_flitter,kernel_size=(1,3),dilation_rate=(1,4))
    x1_PTCN = Reshape((*, *))(x1_PTCN)#Set parameters according to actual conditions
    o_PTCN=SA_PTCN(x1_PTCN)

    o =  Concatenate(axis=1)([o_P, o_H,o_TRIM,o_DRAFT1,o_DRAFT2,o_HW,o_CW,o_PTCN])
    o = Reshape((num_feature, -1))(o)

    GLOAB = G_Attention(num_Loc_att,num_feature,num_g_att)
    GLOAB = GLOAB(o)
    GLOAB = Flatten()(GLOAB)
    out = Dense(1)(GLOAB)
    model = Model(inputs=[input_layer_P,input_layer_H,input_layer_TRIM,input_layer_DRAFT1,input_layer_DRAFT2,input_layer_HW,input_layer_CW,input_layer_PTCN], outputs=out)
    model.summary()
    model.compile(optimizer="adam", loss='mse',metrics=['accuracy'])
    return model
from tensorflow.keras.utils import plot_model
model=get_model()
plot_model(model, to_file='DResLayer_model.png', show_shapes=True)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', verbose=2)
history = model.fit([P_x_train,S_x_train,TRIM_x_train,DRAFT1_x_train,DRAFT2_x_train,HW_x_train,CW_x_train,PTCN_x_train],
                    y_train,
                      batch_size=num_batch,
                      epochs=num_epoch,
                      verbose=2,
                      validation_split=0.1,
                      shuffle=True)
    #迭代图像
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(loss))
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Train and Val Loss')
plt.show()

y_test_predict=model.predict([P_x_test,S_x_test,TRIM_x_test,DRAFT1_x_test,DRAFT2_x_test,HW_x_test,CW_x_test,PTCN_x_test])#预测结果
y_test_predict= mm_y.inverse_transform(y_test_predict)
y_test_inverse=mm_y.inverse_transform(y_test)
xx = easy_result(y_test_inverse, y_test_predict, 'Train', 'TransF', 'longitude')