#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   connector.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/5 9:26 下午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''

from pymongo import MongoClient


class Connector:
    def __init__(self, dbName='', collectionName=''):
        client = MongoClient('localhost', 27017)
        self.db = client[dbName]
        self.collectionName = collectionName
        self.collection = self.db[collectionName]

    def insertOne(self, dic={}):
        insert_result = self.collection.insert_one(dic)
        return insert_result

    def insertMany(self, list=[]):
        insert_result = None
        insert_result = self.collection.insert_many(list)
        return insert_result

    def inserEntity(self,dic={}):
        findResult=self.collection.find(dic)
        if findResult.count() ==0:
            self.collection.insert_one(dic)

    def inserEntity_FB15K237(self,dic={}):
        findResult=self.collection.find({'entityId':dic['entityId']})
        if findResult.count() ==0:
            self.collection.insert_one(dic)
    def inserRelation_FB15K237(self,dic={}):
        findResult=self.collection.find({'relationId':dic['relationId']})
        if findResult.count() ==0:
            self.collection.insert_one(dic)
if __name__ == '__main__':
   pass

 
