#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   statistic.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/12 8:49 上午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''
import numpy as np
from pandas import  DataFrame as df
from data.fb15k237.handleFile import dataProcess
class Statistic(dataProcess):


    def read_dicTypeConstrain(self):
        dicTypeConstrain=np.load('dicTypeConstrain.npy',allow_pickle=True).item()
        for i in dicTypeConstrain.keys():
            headNameList=dicTypeConstrain[i]['headNameList']
            tailNameList=dicTypeConstrain[i]['tailNameList']
            print('%s,headNameList:%s,tailNameList:%s'%(i,len(headNameList),len(tailNameList)))

    def getRelevantTripleNum(self,entityName='',type='asTail'):
        '''
        这是找出以entityName为tailEntity的相关三元组的数量
        :return:
        '''
        if type=='asTail':
            queryKey='tailName'
        else:
            queryKey = 'headName'


        revelantTripleNum=self.Train_connector.collection.find({queryKey:entityName}).count()
        return  revelantTripleNum
    def getRelationCount(self,fileName=''):
        dicRelationNum = {}
        dicTypeConstrain = np.load('dicTypeConstrain.npy', allow_pickle=True).item()
        with open(fileName) as f:
            for line in f.readlines():
                array = line.strip().split('\t')
                relationName = array[1]
                if relationName not in dicRelationNum.keys():
                    dicRelationNum[relationName] = 1
                else:
                    dicRelationNum[relationName] += 1
        data2 = []
        for key in dicRelationNum.keys():
            data2.append([key, dicRelationNum[key],len(dicTypeConstrain[key]['headNameList']),len(dicTypeConstrain[key]['tailNameList'])])
        data2 = df(data2, columns=['relationName', 'relationCount','headNum','tailNum'])
        data2.to_excel('relationCount_%s.xlsx'%fileName)

    def getAverageRelevantTripleNum(self):
        dic={}
        with open('train.txt') as f:
            for line in f.readlines():
                array = line.strip().split('\t')
                headName = array[0]
                tailName = array[2]




                if headName not in dic.keys(): #不在的话，赋值;在的话，没必要再搜一次
                    headNameRelevantTripleNum = self.getRelevantTripleNum(entityName=headName, type='asTail')
                    dic[headName]=headNameRelevantTripleNum
                if tailName not in dic.keys():
                    tailNameRelevantTripleNum = self.getRelevantTripleNum(entityName=tailName, type='asTail')
                    dic[tailName] = tailNameRelevantTripleNum

        data=[]
        for key in dic.keys():
            data.append([key,dic[key]])
        data=df(data,columns=['entityName','relevantTripleNum_asTail'])
        data.to_excel('relevantTripleNum_asTail.xlsx')

    def getHeadNameNum_train(self,relationName='/film/film/release_date_s./film/film_regional_release_date/film_release_region'):
        dicHeadNameNum={}
        result=self.Train_connector.collection.find({'relationName':relationName})
        for line in result:
            headName = line['headName']
            if headName not in dicHeadNameNum.keys():
                dicHeadNameNum[headName]=1
            else:
                dicHeadNameNum[headName]+=1
        data=[]
        for key in dicHeadNameNum.keys():
            data.append([key,dicHeadNameNum[key]])
        data=df(data,columns=['headName','headNameNum'])
        data.to_excel('train_headNameNum.xlsx')
    def getHeadNameNum_valid(self,relationName='/film/film/release_date_s./film/film_regional_release_date/film_release_region'):
        dicHeadNameNum={}
        result=self.Valid_connector.collection.find({'relationName':relationName})
        for line in result:
            headName = line['headName']
            if headName not in dicHeadNameNum.keys():
                dicHeadNameNum[headName]=1
            else:
                dicHeadNameNum[headName]+=1
        data=[]
        for key in dicHeadNameNum.keys():
            occurrentInTrain=self.Train_connector.collection.find(({'headName':key,'relationName':relationName})).count()
            data.append([key,occurrentInTrain,dicHeadNameNum[key]])
        data=df(data,columns=['headName','occurrentInTrain','headNameNum'])
        data.to_excel('valid_headNameNum.xlsx')
if __name__ == '__main__':
    st=Statistic()
    st.getRelationCount('valid.txt')
    #st.getHeadNameNum_valid('/film/film/release_date_s./film/film_regional_release_date/film_release_region')