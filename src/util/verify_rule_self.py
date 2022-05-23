from os import uname_result
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn import svm
import time


def verifier(relation_filepath, iter_num):
    print("第 " + str(iter_num) + " 轮验证开始")
    start_time = time.process_time()
    
    hard_match_result = pd.read_csv(relation_filepath + "5_match_result/hard_match/" + str(iter_num) + ".csv", index_col='sent_id')
    soft_match_result = pd.read_csv(relation_filepath + "5_match_result/soft_match/" + str(iter_num) + ".csv", index_col='sent_id')

    hard_match_result.loc[:, 'human'] = hard_match_result['hard_match']
    soft_match_result.loc[:, 'human'] = soft_match_result['soft_match']

    hard_matched = hard_match_result[hard_match_result['human'] != 0]
    soft_matched = soft_match_result[soft_match_result['human'] != 0]

    # 处理匹配上的
    matched_result = pd.concat([hard_matched, soft_matched])
    matched_result.to_csv(relation_filepath + "1_sent_to_mine/" + str(iter_num + 1) + ".csv", columns=['entity1', 'entity2', 'sent', 'processed_sent', 'human'])

    # 处理没匹配上的
    hard_unmatched = hard_match_result[hard_match_result['human'] == 0]
    soft_unmatched = soft_match_result[soft_match_result['human'] == 0]
    unmatched_result = pd.concat([hard_unmatched, soft_unmatched])
    unmatched_result.to_csv(relation_filepath + "4_sent_to_match/" + str(iter_num + 1) + ".csv", columns=['entity1', 'entity2', 'sent', 'processed_sent', 'human'])

    end_time = time.process_time()
    print("第 " + str(iter_num) + " 轮验证结束，耗时 " + str(end_time - start_time) + " 秒")