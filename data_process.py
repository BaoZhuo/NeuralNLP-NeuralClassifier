# 如果需要使用北大pkuseg分词，通过如下命令安装python包
# !pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pkuseg

"""
数据预处理功能，将csv格式的数据处理成腾讯文本分类包NeuralClassifier的数据格式
之前的数据格式：csv文件，字段为label       item
目标的数据处理格式：
JSON example:
{
    "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
    "doc_token": ["I", "love", "deep", "learning"],
    "doc_keyword": ["deep learning"],
    "doc_topic": ["AI", "Machine learning"]
}
其中doc_keyword和doc_topic可选
author：微信公众号：数据拾光者
date:20200321
"""

import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import codecs
import pkuseg
import sys
import json
import re

df1 = pd.read_csv("data_test.csv", header=None)
df1.columns = ['doc_label', 'doc_token']
df1

"""
data_test.csv中数据格式是这样的：
doc_label 	doc_token
0 	        是要在车里唱歌么？居然还加了隔音棉！

和之前咱们使用bert分类器的数据格式是一样的，之前的是
label       item
"""

# 使用这个文本分类工具，如果是中文的话涉及到分词
# 分词主要用的是jieba分词或者北大pkuseg分词

# 去除停用词
def stop_words(path):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f]

# 使用jieba分词
def tokenize_by_jieba(doc_token):
    seg_list  = jieba.cut(doc_token, cut_all=False)
    return seg_list

# 使用pkuseg分词
# 一次切分一条数据
def tokenize_by_pkuseg(doc_token):
    seg_list = doc_token.split(" ")
    seg_list = pkuseg.pkuseg().cut(doc_token)  # 以默认配置加载模型进行分词
    return seg_list


"""
将dataframe转化成json数据
使用jieba分词
input: dataframe
       数据格式：['doc_label', 'doc_token']
       tokenize_strategy:可以选“jieba”或者“pkuseg”
outout: json数据
{
    "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
    "doc_token": ["I", "love", "deep", "learning"],
    "doc_keyword": ["deep learning"],
    "doc_topic": ["AI", "Machine learning"]
}
"""
def data_process(df, outfile, tokenize_strategy):
    with open(outfile, "w+", encoding='utf-8') as f:
        for indexs in df.index:
            dict1 = {}
            dict1['doc_label'] = [str(df.loc[indexs].values[0])]
            doc_token = df.loc[indexs].values[1]
            # 只保留中文、大小写字母和阿拉伯数字
            #reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            #doc_token =re.sub(reg, '', doc_token)
            print(doc_token)
            # 中文分词
            # 分词策略可以选“jieba”或者“pkuseg”
            if tokenize_strategy=='jieba':
                seg_list = tokenize_by_jieba(doc_token)
            elif tokenize_strategy=='pkuseg':
                seg_list = tokenize_by_pkuseg(doc_token)
            else:
                seg_list = seg_list
                
            # 去除停用词
            content = [x for x in seg_list if x not in stop_words('stop_words.txt')]
            dict1['doc_token'] = content
            dict1['doc_keyword'] = []
            dict1['doc_topic'] = []
            # 组合成字典
            print(dict1)
            # 将字典转化成字符串
            json_str = json.dumps(dict1, ensure_ascii=False)
            # 已添加的方式写入json文件
            f.write('%s\n' % json_str)                       

json_path = 'data_test.json'
# 这里就能将咱们之前的csv文件转换成开源文本分类需要的数据类型了
data_process(df1, json_path, "jieba")      
    

# 上面使用pkuseg分词，但是是单条分词，发现速度有点慢
# 这里咱们使用多条分词，10W条数据量
pkuseg.test("data_test.txt", "data_test_out.txt", nthread=1)
pkuseg.test("data_test2.csv", "data_test_out.csv", nthread=20)
# 数据测试情况
# 20  11.198
# 2   28.542
# 1   44.976

# pkuseg比较特殊
# 这里咱们使用jieba加工模型训练的数据集
# 然后训练模型，看数据处理是否成功
data_process_jiaba(df1, 'train.json')
data_process_jiaba(df1, 'test.json')
data_process_jiaba(df1, 'validate.json')
