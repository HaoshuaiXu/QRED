import pandas as pd
import re
from timeit import default_timer as timer


def hard_match(relation_filepath, iter_num):
    # print("第 " + str(iter_num) + " 轮硬匹配开始")
    print("第 %d 轮硬匹配开始" % iter_num)
    start_time = timer()
    rule_df = pd.read_csv(relation_filepath + "3_original_label_rule/" + str(iter_num) + ".csv", index_col="rule_id")
    sent_df = pd.read_csv(relation_filepath + "4_sent_to_match/" + str(iter_num) + ".csv", index_col="sent_id")
    match_result = match(rule_df, sent_df)
    match_result.to_csv(relation_filepath + "5_match_result/hard_match/" + str(iter_num) + ".csv")
    end_time = timer()
    print("第 %d 轮硬匹配结束，耗时 %.2f 秒，硬匹配数量 %d 条，正例 %d 条，负例 %d 条" \
        %(iter_num, (end_time - start_time), match_result[match_result['hard_match'] != 0].shape[0], match_result[match_result['hard_match'] == 1].shape[0], match_result[match_result['hard_match'] == -1].shape[0]))
    return match_result


def match(rule_df:pd.DataFrame, sent_df:pd.DataFrame):
    sent_df.loc[:, 'hard_match'] = 0
    pos_rule_df = rule_df[rule_df['original_label'] == 1].copy()
    neg_rule_df = rule_df[rule_df['original_label'] == -1].copy()
    for index, row in sent_df.iterrows():
        for pat_str in neg_rule_df['pattern'].tolist():
            match_or_not = re.search(pattern=pat_str, string=row['processed_sent'])
            if match_or_not:
                sent_df.loc[index, 'hard_match'] = -1
    
    for index, row in sent_df[sent_df['hard_match'] == 0].iterrows():
        for pat_str in pos_rule_df['pattern'].tolist():
            match_or_not = re.search(pattern=pat_str, string=row['processed_sent'])
            if match_or_not:
                sent_df.loc[index, 'hard_match'] = 1
    return sent_df


def hard_match_log(match_result:pd.DataFrame, iter_num, relationfilepath):
    pos_match_num = match_result[match_result['hard_match'] == 1].shape[0]
    neg_match_num = match_result[match_result['hard_match'] == -1].shape[0]
    no_match_num = match_result[match_result['hard_match'] == 0].shape[0]
    with open(relationfilepath + 'log/hard_match_log.txt', 'a+') as f:
        f.writelines(str(iter_num) + ',' + str(pos_match_num) + ',' + str(neg_match_num) + ',' + str(no_match_num) + '\n')