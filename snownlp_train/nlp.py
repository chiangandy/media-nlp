# -*- coding: utf-8 -*-
import os
import re
import json
import configparser
import MySQLdb
import time
import datetime
from snownlp import SnowNLP
from snownlp import sentiment
from pymongo import MongoClient
from bson.objectid import ObjectId

# snownlp referneces
# nlp_str = u""" SnowNLP是一個python寫的類庫，可以方便的處理中文文本內容，是受到了TextBlob的啓發而寫的，由於現在大部分的自然語言處理庫基本都是針對英文的，於是寫了一個方便處理中文的類庫，並且和TextBlob不同的是，這裏沒有用NLTK，所有的算法都是自己實現的，並且自帶了一些訓練好的字典。注意本程序都是處理的unicode編碼，所以使用時請自行decode成unicode。 """
# nlp_str = u"""這種情況很難處理 疑點太大 如果誰誰誰向丁特請求道歉 事情就會更複雜 然後輿論就出來了 反而是那位閃問號的人虧損更大了 今天我閃問號 有人幫我出氣要求道歉 結果被做成一支影片成為第二次受傷。 所以你乾脆不要找DINTER道歉 以後也不會有人提及這件事情~~~"""
# nlp_str = u"""沒看卻能知道被嘴，幽默，而且也太自以為，被嗆個幾句在那邊鄭重道歉，我你老師笑到差點從椅子上掉下來"""
# print "s.words"
# print s.words 
# print "s.tags"
# print s.tags
# for aa in s.tags:
#     print aa[0],aa[1]
# for aa in s.keywords(limit=10):
#     print aa[0]    
# print "s.sentiments", s.sentiments
# if s.sentiments >= 0.5:
#     positive = int(((s.sentiments * 100.0) - 50.0) * 2)
#     negative = 0
# else:
#     positive = 0
#     negative = int((50-(s.sentiments * 100.0)) * 2)  #  50.0
# print positive, negative
# print s.tf 
# print s.idf
# Read properties from file
def read_propertis(file_name):
    properties_dict = dict()
    config = configparser.ConfigParser()  
    config.read(file_name)  
    # seg = 'DATABASE'
    seg = 'DATABASE'
    properties_dict['mssql_host'] = config[seg]['mssql_host']
    properties_dict['mssql_username'] = config[seg]['mssql_username']
    properties_dict['mssql_password'] = config[seg]['mssql_password']
    properties_dict['mssql_database'] = config[seg]['mssql_database']
    properties_dict['mssql_port'] = int(config[seg]['mssql_port'])

    seg2 = 'MONGODB'
    properties_dict['mongodb_host'] = config[seg2]['mongodb_host']
    properties_dict['mongodb_username'] = config[seg2]['mongodb_username']
    properties_dict['mongodb_password'] = config[seg2]['mongodb_password']
    properties_dict['mongodb_database'] = config[seg2]['mongodb_collection']
    properties_dict['mongodb_port'] = int(config[seg2]['mongodb_port'])
    return properties_dict

def nlp_get_process():
    db_conf = read_propertis('config.ini')
    db = MySQLdb.connect(host = db_conf['mssql_host'], user = db_conf['mssql_username'], passwd = db_conf['mssql_password'], db = db_conf['mssql_database'], port = db_conf['mssql_port'], charset = 'utf8mb4', use_unicode = True)
    ycursor = db.cursor()
    query_str = """SELECT `comment_text` FROM `channel_comments` limit 500000 """   #limit 50000
    ycursor.execute(query_str)
    results = ycursor.fetchall() 
    result_list = []
    pos = 0
    neg = 0
    cnt = 0
    positive_file = open("src/positive.txt", "w")
    negative_file = open("src/negative.txt", "w")
    for record in results:  
        stmt = record[0].replace('\n','')
        ln = len(stmt)
        if ln <= 150 and ln >= 8:  
            s = SnowNLP(stmt) 
            if s.sentiments > 0.85:
                positive_file.write(stmt.encode('utf8')+'\n')
                # print "positive:", len(stmt), stmt
                pos += 1
            if s.sentiments < 0.15: 
                negative_file.write(stmt.encode('utf8')+'\n')   
                # print "negative:", len(stmt), stmt
                neg += 1  
        cnt += 1
        if (cnt % 100) == 0:
            print "data processed:", cnt            
    db.close()
    positive_file.close()
    negative_file.close()
    print "total_result:", str(pos), str(neg)

def nlp_train_process():
    print "Start training process..."
    # 重新訓練模型
    sentiment.train('src/negative.txt', 'src/positive.txt')
    # 保存模型
    sentiment.save('fino-sentiment.marshal')
    print "End of  training process..."

def do_process():
    print "Start training process..."
    num_recs = 0
    with open('positive.txt','r+') as f:
        for line in f:
            # line = u""" SnowNLP是一個python寫的類庫，可以方便的處理中文文本內容，是受到了TextBlob的啓發而寫的，由於現在大部分的自然語言處理庫基本都是針對英文的，於是寫了一個方便處理中文的類庫，並且和TextBlob不同的是，這裏沒有用NLTK，所有的算法都是自己實現的，並且自帶了一些訓練好的字典。注意本程序都是處理的unicode編碼，所以使用時請自行decode成unicode。 """

            print line.decode('utf8')
            sentence = SnowNLP(line.decode('utf8'))
            words = sentence.words
            for word in words:
                print word
            # if len(words) > maxlen:
            #     maxlen = len(words)
            # for word in words:
            #     word_freqs[word] += 1
            num_recs += 1
            if num_recs % 3 == 0:
                break

def test_process():
    nlp_str = [ u"""沒看卻能知道被嘴，幽默，而且也太自以為，被嗆個幾句在那邊鄭重道歉，我你老師笑到差點從椅子上掉下來""",
                u"""能這樣做已經很好了，不然妳想怎樣""",
                u"""好還要更好，一定要讓你知道""",
                u"""妳的好只有我知道""",
                u"""學點東西就出來現""",
                u"""妳只有這點能耐嗎？""",
                u"""不要再當酸民了，好嗎？""",
                u"""最漂亮的你，依舊動人可愛""",
                u"""妳想我嗎，我很想你""",
                u"""妳的自信讓我發光"""
    ]
    for str in nlp_str:
        s = SnowNLP(str)
        print "score:", "%.5f" % s.sentiments, str

# def nlp_analysis_process():
#     db_conf = read_propertis('config.ini')
#     db = MySQLdb.connect(host = db_conf['mssql_host'], user = db_conf['mssql_username'], passwd = db_conf['mssql_password'], db = db_conf['mssql_database'], port = db_conf['mssql_port'], charset = 'utf8mb4', use_unicode = True)
#     ycursor = db.cursor()
#     query_str = """SELECT `id` FROM `media_channels` where `comment_positive` is null limit 200 """   
#     ycursor.execute(query_str)
#     results = ycursor.fetchall() 
#     result_list = []
#     pos = 0
#     neg = 0
#     positive_file = open("src/positive.txt", "w")
#     negative_file = open("src/negative.txt", "w")

#     for record in results:  
#         query_comment_str = """SELECT `comment_text` FROM `channel_comments` where `channel_id` = '%s' order by update_at desc limit 200 """ % record[0]
#         ycursor.execute(query_comment_str)
#         comment_results = ycursor.fetchall() 
#         cnt = 0
#         senti = 0.0
#         for comment_result in comment_results:
#             s = SnowNLP(comment_result[0]) 
#             senti += s.sentiments
#             cnt += 1
#         if senti > 0.0:    
#             t_senti = senti / cnt
#             if t_senti >= 0.5:
#                 positive = int(((t_senti * 100.0) - 50.0) * 2)
#                 negative = 0
#             else:
#                 positive = 0
#                 negative = int((50-(t_senti * 100.0)) * 2)  #  50.0   
#             print "sentiments_result:",t_senti, positive, negative
#     db.close()

# def mysql_nlp_analysis_process():
#     db_conf = read_propertis('config.ini')
#     db = MySQLdb.connect(host = db_conf['mssql_host'], user = db_conf['mssql_username'], passwd = db_conf['mssql_password'], db = db_conf['mssql_database'], port = db_conf['mssql_port'], charset = 'utf8mb4', use_unicode = True)
#     ycursor = db.cursor()
#     query_str = """SELECT `id` FROM `media_channels` WHERE `comment_positive` is null """   # limit 400 
#     ycursor.execute(query_str)
#     results = ycursor.fetchall() 
#     result_list = []
#     pos = 0
#     neg = 0
#     # positive_file = open("src/positive.txt", "w")
#     # negative_file = open("src/negative.txt", "w")
#     cnt = 0
#     for record in results:  
#         query_comment_str = """SELECT `comment_text` FROM `channel_comments` WHERE `channel_id` = '%s' order by update_at desc limit 200 """ % record[0]
#         ycursor.execute(query_comment_str)
#         comment_results = ycursor.fetchall() 
#         ps_cnt = 0
#         ng_cnt = 0
#         senti = 0.0
#         positive = 0
#         negative = 0
#         for comment_result in comment_results:
#             if len(comment_result[0]) > 8 and len(comment_result[0]) < 200:
#                 try:
#                     s = SnowNLP(comment_result[0]) 
#                     t_senti = s.sentiments
#                     if t_senti >= 0.5:
#                         positive += int(((t_senti * 100.0) - 50.0) * 2)
#                         negative += 0
#                         ps_cnt += 1
#                     else:
#                         positive += 0
#                         negative += int((50-(t_senti * 100.0)) * 2)  #  50.0  
#                         ng_cnt += 1
#                 except:
#                     print "ERROR: process", comment_result[0]        
#         if positive > 0:
#             positive /= ps_cnt   
#         if negative > 0:    
#             negative /= ng_cnt   
#         # print "sentiments:", record[0], positive, negative
#         update_str = """ UPDATE `media_channels` SET `comment_positive` = %s, `comment_negative` = %s WHERE `id` = '%s'
#         """ % (positive, negative, record[0])
#         # print update_str
#         ycursor.execute(update_str)
#         cnt += 1
#         if (cnt % 100) == 0:
#             print "process count", cnt 
#             db.commit()
#     db.commit()
#     db.close()

# def mongodb_nlp_analysis_process():
#     db_conf = read_propertis('config.ini')      
#     client = MongoClient(db_conf['mongodb_host'],db_conf['mongodb_port'])
#     client.the_database.authenticate(db_conf['mongodb_username'], db_conf['mongodb_password'], source='admin', mechanism='SCRAM-SHA-1')
#     db = client.youtube_db
#     media_channels_collection = db['media_channels']
#     channel_comments_collection = db['channel_comments']    
#     channel_lists = media_channels_collection.find( { "comment_positive": { "$exists": False } } )  
#     cnt = 0
#     print media_channels_collection.find( { "comment_positive": { "$exists": False } } ).count()
#     for channel in channel_lists:
#         channel_comment_lists = channel_comments_collection.find( { "channel_id":  channel["pid"]} ) 
#         # print channel["pid"], channel_comment_lists
#         ps_cnt = 0
#         ng_cnt = 0
#         positive = 0
#         negative = 0
#         for comment in channel_comment_lists:
#             if len(comment['comment_text']) > 8 and len(comment['comment_text']) < 200:
#                 try:
#                     s = SnowNLP(comment['comment_text']) 
#                     t_senti = s.sentiments
#                     if t_senti >= 0.5:
#                         positive += int(((t_senti * 100.0) - 50.0) * 2)
#                         negative += 0
#                         ps_cnt += 1
#                     else:
#                         positive += 0
#                         negative += int((50-(t_senti * 100.0)) * 2)  #  50.0  
#                         ng_cnt += 1
#                 except:
#                     print "ERROR: process", comment['comment_text']        
#         if positive > 0:
#             positive /= ps_cnt   
#         if negative > 0:    
#             negative /= ng_cnt   
#         # print "result:", positive, negative
#         media_channels_collection.update_one({
#             'pid': channel["pid"]
#           },{
#             '$set': {
#                 "comment_positive": positive, 
#                 "comment_negative": negative
#             }
#           }, upsert=False)
#         cnt += 1
#         if (cnt % 100) == 0:
#             print "process count", cnt   
#     client.close() 

if __name__ == '__main__':
    # nlp_get_process()
    # nlp_train_process()
    do_process()