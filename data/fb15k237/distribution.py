#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   distribution.py    
@Contact :   Ancientshi@gmail.com
@Modify Time      @Author    @Version    
------------      -------    --------    
2020/12/9 5:18 下午   Ferdinand      1.0  
@Desciption
----------------

----------------     
'''

import numpy as np
def trainDistribution():
    '''
    查看训练数据中，各个relation的情况
    :return:
    '''
    fileNameList = ['train.txt', 'valid.txt', 'test.txt']
    context = ''
    dic = {'train.txt':{},
           'valid.txt':{},
           'test.txt':{}}
    for fileName in fileNameList:
        with open(context + fileName) as f:
            for line in f.readlines():
                array = line.strip().split('\t')
                relationName = array[1]
                if relationName not in dic[fileName].keys():
                    dic[fileName][relationName] = 0
                else:
                    dic[fileName][relationName] += 1
    # for i in dic.keys():
    #     print(i)
    #     for j in dic[i].keys():
    #         print('%s:%s'%(j,dic[i][j]))
    dic=np.array(dic)
    np.save('relationDistribution.npy',dic)
if __name__ == '__main__':
    trainDistribution()