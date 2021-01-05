#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   convCFKG.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/5 6:34 下午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''
import numpy as np
import sklearn
import tensorflow as tf
from keras import losses
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
# 构建TextCNN模型
from tensorflow.python.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, Dropout, Dense, \
    Activation, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model
from keras.optimizers import Adam

from myCode.connector import Connector
from experimentToRecord.evaluate_myConvKB_type13 import Recommend
import math
connectorEntity = Connector(dbName='meta_data', collectionName='entityIdAndType')
def myLoss(y_true, y_pred):
    # lossSum = tf.Variable(0.0, dtype=tf.float32)
    value1 = tf.multiply(y_true, y_pred)
    value2=tf.math.exp(value1)
    value3 = tf.add(value2, 1)
    value4=tf.math.log(value3)
    len=tf.cast(tf.shape(value4)[0],tf.float32)
    lossSum=tf.reduce_sum(value4)

    avgLoss = tf.divide(lossSum, len)




    return avgLoss
    # log = tf.math.log(b)
    # lenth=y_true.get_shape().as_list()
    # # lossSum = tf.Variable(0.0, dtype=tf.float32)
    # # shape=tf.shape(y_true)
    #
    # y_true=y_true.numpy()
    # y_pred=y_pred.numpy()
    # for i in range(len(y_true)):
    #     if y_true[i]==0:
    #         l=-1
    #     else:
    #         l=1
    #     lossSum+=math.log(1+math.exp(l*y_pred[i]))
    # lossSum = tf.constant(lossSum, dtype=tf.dtypes.float32)
    # return lossSum
def model1(x_train, y_train, x_test, y_test):
    #200维度的三元组（head,relation,tail）
    inputs = keras.Input(shape=(100,3,1,))
    cnn1 = Conv2D(filters=50,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_size=(1,3), padding='valid', strides=1, activation='relu')(inputs)
    flat = Flatten()(cnn1)

    drop = Dropout(0.2)(flat)
    #out1 = Dense(units=1,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0005))(drop)
    out1 = Dense(units=1,use_bias=False)(drop)

    # net
    model1 = Model(inputs, out1)
    #model1.compile()
    #model1.summary()
    #ou1_output=model1.predict(x_train,) #shape为(-1,1)
    model1.compile(loss=myLoss,optimizer=Adam(6e-6))
    model1.summary()
    loss_value = []
    history=model1.fit(x_train, y_train, batch_size=30, epochs=200,validation_data=(x_test, y_test))
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='valid')
    pyplot.legend()
    pyplot.show()


    model1.save('../data/modelFile/originalConvKB_onlyType13_%s_%s.h5' %
                (round(history.history['loss'][-1],4),round(history.history['val_loss'][-1],4)))
    # result = model1.predict(x_test)  # 预测样本属于每个类别的概率
    # print(result)


def model2(x_train, y_train, x_test, y_test,model1):
    model2 = Model(inputs=model1.input, outputs=model1.output)
    model1_output=model2.predict(x_train)
    print(model1_output)
    # model.compile(loss=losses.binary_crossentropy,optimizer=Adam(6e-6), metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Recall()])
    #
    # model.summary()
    # print(model.layers[4].output)
    # model.fit(x_train, y_train, batch_size=30, epochs=50)
    # # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    # result = model.predict(x_test)  # 预测样本属于每个类别的概率
    # y_predict=[]
    # for r in result:
    #     if r[0]>=0.5:
    #         y_predict.append(1)
    #     else:
    #         y_predict.append(0)
    # acu=sklearn.metrics.accuracy_score(y_test, y_predict)
    # f1=sklearn.metrics.f1_score(y_test, y_predict, average='weighted')
    # print('准确率', acu)
    # print('平均f1-score:',f1)
    # model.save('../data/modelFile/convCFKG_onlyOneType13_BCE_evaluate%s.h5'%f1)
    #
    # rm=Recommend()
    # rm.setModel('../data/modelFile/convCFKG_onlyOneType13_BCE_evaluate%s.h5'%f1)
    # rm.caculateHitRation()
    # rm.caculateMRandMRR()




if __name__ == '__main__':
    def transY(array=[]):
        y=[]
        for i in array:
            if i==0:
                y.append(-1)
            else:
                y.append(1)
        return np.array(y,dtype=float)
    #formFeedData()
    feedModelDic=np.load('../data/npy/feedModelDic_onlyOneType_13.npy',allow_pickle=True)
    x_train_embedding= feedModelDic.item()['x_train_embedding']
    y_train= transY(feedModelDic.item()['y_train'])

    x_valid_embedding= feedModelDic.item()['x_valid_embedding']
    y_valid= transY(feedModelDic.item()['y_valid'])
    x_test_embedding= feedModelDic.item()['x_test_embedding']
    y_test= transY(feedModelDic.item()['y_test'])
    model1=model1(x_train_embedding.reshape(-1,100,3), y_train, x_valid_embedding.reshape(-1,100,3), y_valid)
    #model2(x_train_embedding.reshape(-1,100,3), y_train, x_valid_embedding.reshape(-1,100,3), y_valid,model1)
    # convCFKG_model(x_train_embedding.reshape(-1,100,3), y_train, x_valid_embedding.reshape(-1,100,3), y_valid)

