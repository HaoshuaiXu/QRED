import pandas as pd
from gensim.models import Word2Vec
from timeit import default_timer as timer


def soft_match(relation_filepath, iter_num, threshold):
    # print("第 " + str(iter_num) + " 轮软匹配开始")
    print("第 %d 轮软匹配开始" % iter_num)
    start_time = timer()
    rule_df = pd.read_csv(relation_filepath + "3_original_label_rule/" + str(iter_num) + ".csv", index_col='rule_id')
    sent_df = pd.read_csv(relation_filepath + "4_sent_to_match/" + str(iter_num) + ".csv", index_col='sent_id')
    match_result = match(rule_df, sent_df, threshold)
    match_result.to_csv(relation_filepath + "5_match_result/soft_match/" + str(iter_num) + ".csv")
    end_time = timer()
    print("第 %d 轮软匹配结束，耗时 %.2f 秒，软匹配数量 %d 条，正例 %d 条，负例 %d 条" \
        %(iter_num, (end_time - start_time), match_result[match_result['soft_match'] != 0].shape[0], match_result[match_result['soft_match'] == 1].shape[0], match_result[match_result['soft_match'] == -1].shape[0]))
    return match_result


def match(rule_df:pd.DataFrame, sent_df:pd.DataFrame, threshold=0.7):
    sent_df.loc[:, 'soft_match_score'] = 0
    sent_df.loc[:, 'soft_match'] = 0
    neg_rule_list = rule_df[rule_df['original_label'] == -1]['rule'].tolist()
    pos_rule_list = rule_df[rule_df['original_label'] == 1]['rule'].tolist()
    encoder = Word2Vec.load("/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/word2vec/word2vec.model")
    score_list = []
    if len(neg_rule_list) == 0:
        pass
    else:
        for index, row in sent_df.iterrows():
            sent_word_seq = row['processed_sent'].split()
            soft_match_score_sum = 0
            for rule in neg_rule_list:
                rule_word_seq = rule.split('/')
                soft_match_score = encoder.wv.n_similarity(sent_word_seq, rule_word_seq)
                soft_match_score_sum = soft_match_score_sum + soft_match_score
            soft_match_score_aver = soft_match_score_sum / len(neg_rule_list) # 求平均数
            score_list.append(soft_match_score_aver)
        sent_df.loc[:, 'soft_match_score'] = score_list
        sent_df.loc[sent_df['soft_match_score'] >= threshold, 'soft_match'] = -1
    
    if len(pos_rule_list) == 0:
        return sent_df
    else:
        score_list = []
        for index, row in sent_df[sent_df['soft_match'] == 0].iterrows():
            sent_word_seq = row['processed_sent'].split()
            soft_match_score_sum = 0
            for rule in pos_rule_list:
                rule_word_seq = rule.split('/')
                soft_match_score = encoder.wv.n_similarity(sent_word_seq, rule_word_seq)
                soft_match_score_sum = soft_match_score_sum + soft_match_score
            soft_match_score_aver = soft_match_score_sum / len(pos_rule_list) # 求平均数
            score_list.append(soft_match_score_aver)
        sent_df.loc[sent_df['soft_match'] == 0, 'soft_match_score'] = score_list
        sent_df.loc[(sent_df['soft_match_score'] >= threshold) & (sent_df['soft_match'] == 0), 'soft_match'] = 1

        return sent_df