import os
import pandas as pd
from timeit import default_timer as timer


def merge_unmatched_sent(relation_filepath, iter_num, label_prop):
    if label_prop == 1 or label_prop == 0:
        print("第 %d 轮合并未标注句子开始" % iter_num)
        start_time = timer()
        hard_match_result = pd.read_csv(relation_filepath + "5_match_result/hard_match/" + str(iter_num) + ".csv", index_col='sent_id', usecols=['sent_id', 'entity1', 'entity2', 'sent', 'processed_sent', 'label', 'hard_match'])
        soft_match_result = pd.read_csv(relation_filepath + "5_match_result/soft_match/" + str(iter_num) + ".csv", index_col='sent_id', usecols=['sent_id', 'entity1', 'entity2', 'sent', 'processed_sent', 'label', 'soft_match'])
        hard_unmatched = hard_match_result[hard_match_result['hard_match'] == 0]
        soft_unmatched = soft_match_result[soft_match_result['soft_match'] == 0]
        hard_unmatched = hard_unmatched[['entity1', 'entity2', 'sent', 'processed_sent', 'label']]
        soft_unmatched = soft_unmatched[['entity1', 'entity2', 'sent', 'processed_sent', 'label']]
        result = pd.concat([hard_unmatched, soft_unmatched])
        result.drop_duplicates('sent', inplace=True)
        result.to_csv(relation_filepath + "4_sent_to_match/" + str(iter_num + 1) + ".csv")
        end_time = timer()
        print("第 %d 轮未标注句子结束，耗时 %.2f 秒" % (iter_num, (end_time - start_time)))
    else:
        print("第 %d 轮合并未标注句子开始" % iter_num)
        start_time = timer()
        hard_match_result = pd.read_csv(relation_filepath + "5_match_result/hard_match/" + str(iter_num) + ".csv", index_col='sent_id', usecols=['sent_id', 'entity1', 'entity2', 'sent', 'processed_sent', 'label', 'hard_match'])
        soft_match_result = pd.read_csv(relation_filepath + "5_match_result/soft_match/" + str(iter_num) + ".csv", index_col='sent_id', usecols=['sent_id', 'entity1', 'entity2', 'sent', 'processed_sent', 'label', 'soft_match'])
        un_sample = pd.read_csv(os.path.join(os.path.join(relation_filepath, '8_unsample', str(iter_num), '.csv')), index_col='sent_id')
        hard_unmatched = hard_match_result[hard_match_result['hard_match'] == 0]
        soft_unmatched = soft_match_result[soft_match_result['soft_match'] == 0]
        hard_unmatched = hard_unmatched[['entity1', 'entity2', 'sent', 'processed_sent', 'label']]
        soft_unmatched = soft_unmatched[['entity1', 'entity2', 'sent', 'processed_sent', 'label']]
        result = pd.concat([hard_unmatched, soft_unmatched, un_sample])
        result.drop_duplicates('sent', inplace=True)
        result.to_csv(os.path.join(relation_filepath, "4_sent_to_match/", str(iter_num + 1), ".csv"))
        end_time = timer()
        print("第 %d 轮未标注句子结束，耗时 %.2f 秒" % (iter_num, (end_time - start_time)))