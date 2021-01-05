#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   class handleFile.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/9 11:59 上午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''
import random

import numpy as np

from myCode.connector import Connector

class dataProcess:
    entityNum=14541
    relationNum=237
    trainNum=0
    validNum=0
    testNum=0
    vecDim=100
    Entity2id_connector=None
    Relation2id_connector=None
    context=''
    dicTypeConstrain={}
    def __init__(self):
        self.Entity2id_connector=Connector('FB15K237','entity2id')
        self.Relation2id_connector = Connector('FB15K237','relation2id')
        self.Train_connector = Connector('FB15K237','train')
        self.Valid_connector = Connector('FB15K237','valid')
        self.Test_connector = Connector('FB15K237','test')
    def loadDicTypeConstrain(self):
        self.dicTypeConstrain = np.load(self.context + 'dicTypeConstrain.npy', allow_pickle=True).item()

    def read_train(self):
        '''
        *****已初始化
        初始化一次，数据插入到数据库，建立索引，再运行一次
        :return:
        '''
        fileNameList = ['train.txt']
        context = ''
        for fileName in fileNameList:

            x = []
            y = []
            with open(context + fileName) as f:
                times = 0
                for line in f.readlines():
                    array = line.strip().split('\t')
                    headName = array[0]
                    tailName = array[2]
                    relationName = array[1]
                    self.Train_connector.insertOne({'headName':headName,
                                                    'tailName':tailName,
                                                    'relationName':relationName})
                    if times % 2000 == 0:
                        print('%s已经%s次' % (fileName, times))
                    times += 1
    def read_valid(self):
        '''
        *****已初始化
        初始化一次，数据插入到数据库，建立索引，再运行一次
        :return:
        '''
        fileNameList = ['valid.txt']
        context = ''
        for fileName in fileNameList:

            x = []
            y = []
            with open(context + fileName) as f:
                times = 0
                for line in f.readlines():
                    array = line.strip().split('\t')
                    headName = array[0]
                    tailName = array[2]
                    relationName = array[1]
                    self.Valid_connector.insertOne({'headName':headName,
                                                    'tailName':tailName,
                                                    'relationName':relationName})
                    if times % 2000 == 0:
                        print('%s已经%s次' % (fileName, times))
                    times += 1
    def read_test(self):
        '''
        *****已初始化
        初始化一次，数据插入到数据库，建立索引，再运行一次
        :return:
        '''
        fileNameList = ['test.txt']
        context = ''
        for fileName in fileNameList:

            x = []
            y = []
            with open(context + fileName) as f:
                times = 0
                for line in f.readlines():
                    array = line.strip().split('\t')
                    headName = array[0]
                    tailName = array[2]
                    relationName = array[1]
                    self.Test_connector.insertOne({'headName':headName,
                                                    'tailName':tailName,
                                                    'relationName':relationName})
                    if times % 2000 == 0:
                        print('%s已经%s次' % (fileName, times))
                    times += 1

    def read_entity2id_relation2id(self):
        '''
        *****已初始化
        初始化一次，数据插入到数据库，建立索引，再运行一次
        entity2id_relation2id 分隔符为\t entityName entityId
        :return:
        '''
        entity2idfileName='entity2id.txt'
        entity2vec100='entity2vec100.init'
        relation2idfileName='relation2id.txt'
        relation2vec100='relation2vec100.init'
        context=''
        with open(context+entity2idfileName) as file1:
            with open(context+entity2vec100) as file2:
                for (line1,line2) in zip(file1.readlines(),file2.readlines()):
                    array = line1.strip().split('\t')
                    entityName=array[0]
                    entityId=array[1]

                    Embedding=line2.strip()
                    dic={'entityName':entityName,
                         'entityId':int(entityId),
                         'Embedding':Embedding}
                    self.Entity2id_connector.inserEntity_FB15K237(dic)
                
        with open(context+relation2idfileName) as file1:
            with open(context+relation2vec100) as file2:
                for (line1,line2) in zip(file1.readlines(),file2.readlines()):
                    array = line1.strip().split('\t')
                    relationName=array[0]
                    relationId=array[1]

                    Embedding=line2.strip()
                    dic={'relationName':relationName,
                         'relationId':int(relationId),
                         'Embedding':Embedding}
                    self.Relation2id_connector.inserRelation_FB15K237(dic)
    def generateTypeConstrain(self):
        '''
        *****已初始化
        初始化一次，生成npy文件即可
        从train中形成类型约束,dic dic List类型
        :return:
        '''
        fileName='train.txt'
        context=''
        dic={}
        with open(context+fileName) as f:
            for line in f.readlines():
                array=line.strip().split('\t')
                headName=array[0]
                tailName=array[2]
                relationName=array[1]
                #如果是第一次遇到新的关系，初始化
                if relationName not in dic.keys():
                    dic[relationName]={'headNameList':[],
                                       'tailNameList':[]}
                #那么就是关系已经在里面了，可以用键访问
                else:
                    #如果headName,tailName不在list，加入；如果在，就不用再加了
                    if headName not in dic[relationName]['headNameList']:
                        dic[relationName]['headNameList'].append(headName)
                    if tailName not in dic[relationName]['tailNameList']:
                        dic[relationName]['tailNameList'].append(tailName)
        dicArray=np.array(dic)
        np.save(context+'dicTypeConstrain',dicArray)

    def getEmbeddingByName(self,name='',type=''):
        vec = []
        if type=='entity':

            result=self.Entity2id_connector.collection.find({'entityName':name})
            for i in result:
                for numStr in i['Embedding'].strip().split('\t'):
                    vec.append(float(numStr))
        elif type=='relation':
            result=self.Relation2id_connector.collection.find({'relationName':name})
            for i in result:
                for numStr in i['Embedding'].strip().split('\t'):
                    vec.append(float(numStr))
        return vec
    def getIdByName(self,name='',type=''):
        id=None
        if type=='entity':

            result=self.Entity2id_connector.collection.find(query={'entityName':name})
            for i in result:
                id=int(i['entityId'])
        elif type=='relation':
            result=self.Relation2id_connector.collection.find(query={'relationName':name})
            for i in result:
                id = int(i['relationId'])
        return id
    def generateInvalidTriple(self,headName='',tailName='',relationName=''):
        tailNameList=self.dicTypeConstrain[relationName]['tailNameList'][:]
        if tailName in tailNameList:
            tailNameList.remove(tailName)

        #为了不随机生成train.txt中存在的数据
        result=self.Train_connector.collection.find({'headName':headName,
                                              'relationName':relationName})
        for r in result:
            try:
                tailNameList.remove(r['tailName'])
            except:
                pass

        #如果只有一个候选的tail(已经移除)，那么返回0000
        if len(tailNameList) ==0 :
            headEmbedding = np.zeros((1,100))[0]
            tailEmbedding = np.zeros((1,100))[0]
            relationEmbedding = np.zeros((1,100))[0]
            return [headEmbedding, tailEmbedding, relationEmbedding]
        else:

            randomNum=random.randint(0, len(tailNameList) - 1)
            randomTailName=tailNameList[randomNum]

            headEmbedding = self.getEmbeddingByName(name=headName, type='entity')
            tailEmbedding = self.getEmbeddingByName(name=randomTailName, type='entity')
            relationEmbedding = self.getEmbeddingByName(name=relationName, type='relation')

            return [headEmbedding,tailEmbedding,relationEmbedding]

    def read_train_valid_test(self):
        '''
        Train_valid_test 分隔符为\t headName  relationName tailName
        :return:
        '''
        fileNameList=['valid.txt','train.txt','test.txt']
        context=''
        feedModelDic={}

        for fileName in fileNameList:

            x = []
            y = []
            with open(context+fileName) as f:
                times = 0
                for line in f.readlines():
                    array=line.strip().split('\t')
                    headName=array[0]
                    tailName=array[2]
                    relationName=array[1]
                    headEmbedding=self.getEmbeddingByName(name=headName,type='entity')
                    tailEmbedding = self.getEmbeddingByName(name=tailName, type='entity')
                    relationEmbedding = self.getEmbeddingByName(name=relationName, type='relation')

                    #有效的三元组
                    x.append([headEmbedding,tailEmbedding,relationEmbedding])
                    y.append(1)

                    #生成被破坏的三元组,改变的是tail
                    invalidTripleEmbedding=self.generateInvalidTriple(headName=headName,tailName=tailName,relationName=relationName)
                    x.append(invalidTripleEmbedding)
                    y.append(0)
                    if times % 2000 == 0:
                        print('%s已经%s次' % (fileName, times))
                    times += 1
            x=np.array(x)
            y=np.array(y)
            feedModelDic['x_%s_embedding'%fileName[:-4]]=x
            feedModelDic['y_%s'%fileName[:-4]]=y

        feedModelDic=np.array(feedModelDic)
        np.save(self.context+'feedModelDic.npy',feedModelDic)


    def filterOnlyOneType_validData(self):
        typeName='/film/film/release_date_s./film/film_regional_release_date/film_release_region'
        context=''
        fileName='test.txt'
        validList=[]
        with open(context+fileName) as file:
            for line in file.readlines():
                array = line.strip().split('\t')
                relationName = array[1]
                if relationName != '/film/film/release_date_s./film/film_regional_release_date/film_release_region':
                    continue
                validList.append(line)
        validList=np.array(validList)
        np.save('../npy/test2id_type13.npy',validList)

    def read_train_valid_test_onlyOneType(self,theRelation='',relationId=0):
        '''
        Train_valid_test 分隔符为\t headName  relationName tailName
        :return:
        '''
        fileNameList=['train.txt','valid.txt','test.txt']
        context=''
        feedModelDic={}

        for fileName in fileNameList:

            x = []
            y = []
            with open(context+fileName) as f:
                times = 0
                for line in f.readlines():

                    array=line.strip().split('\t')
                    headName=array[0]
                    tailName=array[2]
                    relationName=array[1]
                    if relationName != theRelation:
                        continue
                    headEmbedding=self.getEmbeddingByName(name=headName,type='entity')
                    tailEmbedding = self.getEmbeddingByName(name=tailName, type='entity')
                    relationEmbedding = self.getEmbeddingByName(name=relationName, type='relation')

                    #有效的三元组
                    x.append([headEmbedding,tailEmbedding,relationEmbedding])
                    y.append(1)

                    #生成被破坏的三元组,改变的是tail
                    invalidTripleEmbedding=self.generateInvalidTriple(headName=headName,tailName=tailName,relationName=relationName)
                    x.append(invalidTripleEmbedding)
                    y.append(0)
                    if times % 2000 == 0:
                        print('%s已经%s次' % (fileName, times))
                    times += 1
            x=np.array(x)
            y=np.array(y)
            feedModelDic['x_%s_embedding'%fileName[:-4]]=x
            feedModelDic['y_%s'%fileName[:-4]]=y

        feedModelDic=np.array(feedModelDic)
        np.save(self.context+'../npy/feedModelDic_onlyOneType_%s.npy'%relationId,feedModelDic)

if __name__ == '__main__':
    dp=dataProcess()
    #dp.read_valid()
    #dp.read_test()
    dp.loadDicTypeConstrain()
    dp.read_train_valid_test_onlyOneType('/award/award_nominee/award_nominations./award/award_nomination/award',19)
    #dp.read_train_valid_test_onlyOneType()
 
