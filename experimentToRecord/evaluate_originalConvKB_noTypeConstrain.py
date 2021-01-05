#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate_transE_type13.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/12 3:49 下午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   recommend.py
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version
------------      -------    --------
2020/12/9 2:43 下午   Ferdinand      1.0
@Desciption
----------------

----------------
'''
import random

import keras
import numpy as np
from data.fb15k237.handleFile import dataProcess

import tensorflow as tf
def myLoss(y_true, y_pred):
    # lossSum = tf.Variable(0.0, dtype=tf.float32)
    value1 = tf.multiply(y_true, y_pred)
    value2 = tf.math.exp(value1)
    value3 = tf.add(value2, 1)
    value4 = tf.math.log(value3)
    lossSum = tf.reduce_sum(value4)
    return lossSum
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
class Recommend():

    def __init__(self):
        self.df = dataProcess()
        self.df.context='../data/fb15k237/'
        self.df.loadDicTypeConstrain()
    def setModel(self,modelPath=''):
        self.model = keras.models.load_model(modelPath,custom_objects={'myLoss': myLoss})
        self.model.summary()

    def topN(self, predictResult, n, trueIndex):
        '''

        :param predictResult: 预测概率列表
        :param n: topN,N=?
        :param productsIds: 用户真实购买列表
        :param trueIndex: 为0
        :return:
        '''
        # 概率前n大
        result_label = np.argpartition(predictResult, n)[:n]
        if trueIndex in result_label:
            return 1
        else:
            return 0



    def caculateMR(self,probabilityList=[],appendSign=0):
        validProbability=probabilityList[appendSign]
        biggerNum=0
        for i in probabilityList:
            if i<validProbability:
                biggerNum+=1
        return biggerNum+1
    def getMR(self,headName='',tailName='',relationName=''):
        headEmbedding = self.df.getEmbeddingByName(name=headName,type='entity')
        tailEmbedding = self.df.getEmbeddingByName(name=tailName,type='entity')
        relationEmbedding = self.df.getEmbeddingByName(name=relationName,type='relation')
        x_valid = []
        allRevelantTails=self.df.dicTypeConstrain[relationName]['tailNameList'][:]
        try: #怕玩意里面有
            allRevelantTails.remove(tailName)
        except:
            pass
        for otherTail in allRevelantTails:
            otherTailEmbedding = self.df.getEmbeddingByName(name=otherTail, type='entity')
            x_valid.append([headEmbedding,otherTailEmbedding,relationEmbedding])
        # 再把正确的三元组加入
        try:
            appendSign = random.randint(0, len(x_valid) - 1)
        except:
            return 0,0
        x_valid.insert(appendSign, [headEmbedding, tailEmbedding, relationEmbedding])
        x_valid = np.array(x_valid).reshape(len(x_valid), 100, 3, 1)
        disctance = self.model.predict(x_valid)  # 这里是概率，越大越好
        rank=self.caculateMR(disctance,appendSign)
        return rank,len(disctance)

    def getHit(self,headName='',tailName='',relationName=''):
        testSize=100
        headEmbedding = self.df.getEmbeddingByName(name=headName,type='entity')
        tailEmbedding = self.df.getEmbeddingByName(name=tailName,type='entity')
        relationEmbedding = self.df.getEmbeddingByName(name=relationName,type='relation')
        x_valid = []
        while len(x_valid) is not testSize-1:
            # 生成被破坏的三元组,改变的是tail
            invalidTripleEmbedding = self.df.generateInvalidTriple(headName=headName, tailName=tailName,
                                                                   relationName=relationName)
            x_valid.append(invalidTripleEmbedding)
        # 再把正确的三元组加入
        appendSign = random.randint(0, testSize - 1)
        x_valid.insert(appendSign, [headEmbedding, tailEmbedding, relationEmbedding])
        x_valid = np.array(x_valid).reshape(testSize, 100, 3, 1)
        result = self.model.predict(x_valid)  # 预测样本属于每个类别的概率,result的shape应该是（len(entityEmbedding),2）

        buyProbability = np.array(result).reshape(1,testSize)[0].tolist()
        # hit==1表示topN中有，hit==1表示没有
        TopN = [1, 3, 10]
        hit1 = self.topN(buyProbability, TopN[0], appendSign)
        hit3 = self.topN(buyProbability, TopN[1], appendSign)
        hit10 = self.topN(buyProbability, TopN[2], appendSign)

        return hit1, hit3, hit10

    def caculateHitRation(self):
            '''
            用于形成convCFKG的验证数据
            格式为
            :return:
            数据格式为字典：
             '94474': ['4033', '8435', '22442'],
             key为user实体，value是product实体list
            '''

            times = 0
            hitNum_1 = 0
            hitNum_3 = 0
            hitNum_10 = 0
            with open('../data/fb15k237/valid.txt') as f:

                for line in f.readlines():
                    array = line.strip().split()
                    headName = array[0]
                    tailName = array[2]
                    relationName = array[1]
                    if relationName!='/film/film/release_date_s./film/film_regional_release_date/film_release_region':
                        continue

                    hit_num1, hit_num3, hit_num10 = self.getHit(headName, tailName, relationName)
                    hitNum_1 += hit_num1
                    hitNum_3 += hit_num3
                    hitNum_10 += hit_num10
                    times += 1
                    if times % 1 == 0:
                        print('-' * 20)
                        print('productNum:%s' % times)
                        print('hit@1:%s' % (hitNum_1 / times))
                        print('hit@3:%s' % (hitNum_3 / times))
                        print('hit@10:%s' % (hitNum_10 / times))
                        print('-' * 20)
                        # break
    def caculateMRandMRR_limit(self):
            '''
            用于形成convCFKG的验证数据
            格式为
            :return:
            数据格式为字典：
             '94474': ['4033', '8435', '22442'],
             key为user实体，value是product实体list
            '''
            relationDistribution=np.load('../data/npy/relationDistribution.npy',allow_pickle=True).item()
            relationDistributionInValid=relationDistribution['valid.txt']
            relationCount={}
            times = 0
            RANK=0
            R_RANK=0
            SETSIZE=0
            dicLimit={}
            with open('../data/fb15k237/valid.txt') as f:


                for line in f.readlines():
                    array=line.strip().split()
                    headName=array[0]
                    tailName=array[2]
                    relationName=array[1]
                    if relationName not in dicLimit:
                        dicLimit[relationName] = 1
                    else:
                        if dicLimit[relationName] >= 5:
                            continue
                        else:
                            dicLimit[relationName] += 1

                    rank,setSize=self.getMR(headName,tailName,relationName)
                    if rank==0 and setSize==0:
                        continue
                    print('----%s    %s----'%(rank,setSize))
                    SETSIZE+=setSize
                    RANK+=rank
                    R_RANK+=1/rank
                    times += 1
                    if times % 100 == 0:
                        print('-' * 20)
                        print('productNum:%s' % times)
                        print('MR:%s' % int(RANK/times))
                        print('MRR:%s' % (R_RANK/times))
                        print('MsetSize:%s' % (SETSIZE/times))
                        print('-' * 20)
                        #break
    def caculateMRandMRR(self):
        '''
        用于形成convCFKG的验证数据
        格式为
        :return:
        数据格式为字典：
         '94474': ['4033', '8435', '22442'],
         key为user实体，value是product实体list
        '''
        times = 0
        RANK = 0
        R_RANK = 0
        SETSIZE = 0
        with open('../data/fb15k237/valid.txt') as f:

            for line in f.readlines():
                array = line.strip().split()
                headName = array[0]
                tailName = array[2]
                relationName = array[1]
                if relationName != '/film/film/release_date_s./film/film_regional_release_date/film_release_region':
                    continue
                rank, setSize = self.getMR(headName, tailName, relationName)
                print('-----%s   %s------' % (rank, setSize))
                if rank == 0 and setSize == 0:
                    continue
                SETSIZE += setSize
                RANK += rank
                R_RANK += 1 / rank
                times += 1
                if times % 1 == 0:
                    print('-' * 20)
                    print('productNum:%s' % times)
                    print('MR:%s' % int(RANK / times))
                    print('MRR:%s' % (R_RANK / times))
                    print('MsetSize:%s' % (SETSIZE / times))
                    print('-' * 20)
                    # break
    def caculateHitRation_limit(self):
            '''
            用于形成convCFKG的验证数据
            格式为
            :return:
            数据格式为字典：
             '94474': ['4033', '8435', '22442'],
             key为user实体，value是product实体list
            '''

            times = 0
            hitNum_1 = 0
            hitNum_3 = 0
            hitNum_10 = 0
            dicLimit={}
            with open('../data/fb15k237/valid.txt') as f:

                for line in f.readlines():
                    array = line.strip().split()
                    headName = array[0]
                    tailName = array[2]
                    relationName = array[1]
                    if relationName not in dicLimit:
                        dicLimit[relationName]=1
                    else:
                        if dicLimit[relationName]>3:
                            continue
                        else:
                            dicLimit[relationName] += 1

                    hit_num1, hit_num3, hit_num10 = self.getHit(headName, tailName, relationName)
                    hitNum_1 += hit_num1
                    hitNum_3 += hit_num3
                    hitNum_10 += hit_num10
                    times += 1
                    if times % 1 == 0:
                        print('-' * 20)
                        print('productNum:%s' % times)
                        print('hit@1:%s' % (hitNum_1 / times))
                        print('hit@3:%s' % (hitNum_3 / times))
                        print('hit@10:%s' % (hitNum_10 / times))
                        print('-' * 20)
                        # break


if __name__ == '__main__':
    rm=Recommend()
    rm.setModel('../data/modelFile/originalConvKB_noTypeConstrain_0.6315_0.6149.h5')
    #rm.caculateHitRation_limit()
    rm.caculateMRandMRR_limit()
