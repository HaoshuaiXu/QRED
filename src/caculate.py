import pandas as pd
from gensim.models import Word2Vec
# from gensim.models.fasttext import FastText
import re


def caculate(relation_filepath):
    # relation_filepath = "/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/relation_data/couple/"
    for iter_num in range(1, 17):

        valid_df = pd.read_csv(relation_filepath + "validation/validation_data.csv", index_col='sent_id')
        rule_df = pd.read_csv(relation_filepath + "rule_set/" + str(iter_num) + ".csv")

        pos_rule_num = rule_df[rule_df['final_label'] == 1].shape[0]
        neg_rule_num = rule_df[rule_df['final_label'] == -1].shape[0]


        threshold = 0.7

        # 负规则 硬匹配
        valid_df.loc[:, 'match'] = 0
        neg_rule_df = rule_df[rule_df['final_label'] == -1].copy()
        if neg_rule_df.shape[0] == 0:
            pass
        else:
            for index, row in valid_df.iterrows():
                for pat_str in neg_rule_df['pattern'].tolist():
                    match_or_not = re.search(pattern=pat_str, string=row['processed_sent'])
                    if match_or_not:
                        valid_df.loc[index, 'match'] = -1

        # 负规则 软匹配
        valid_df = valid_df[valid_df['match'] == 0].copy()
        valid_df.loc[:, 'soft_match_score'] = 0
        if neg_rule_df.shape[0] == 0:
            pass
        else:
            neg_rule_list = neg_rule_df.loc[:, 'rule'].tolist()
            encoder = Word2Vec.load("/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/word2vec/word2vec.model")
            # encoder = FastText.load("/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/fasttext/fasttext.model")
            score_list = []
            if len(neg_rule_list) == 0:
                pass
            else:
                for index, row in valid_df.iterrows():
                    sent_word_seq = row['processed_sent'].split()
                    soft_match_score_sum = 0
                    for rule in neg_rule_list:
                        rule_word_seq = rule.split('/')
                        soft_match_score = encoder.wv.n_similarity(sent_word_seq, rule_word_seq)
                        soft_match_score_sum = soft_match_score_sum + soft_match_score
                    soft_match_score_aver = soft_match_score_sum / len(neg_rule_list) # 求平均数
                    score_list.append(soft_match_score_aver)
                valid_df.loc[:, 'soft_match_score'] = score_list
                valid_df.loc[valid_df['soft_match_score'] >= threshold, 'match'] = -1

        # 正规则 硬匹配
        valid_df_left = valid_df[valid_df['match'] == 0].copy()
        pos_rule_df = rule_df[rule_df['final_label'] == 1].copy()
        for index, row in valid_df_left.iterrows():
            for pat_str in neg_rule_df['pattern'].tolist():
                match_or_not = re.search(pattern=pat_str, string=row['processed_sent'])
                if match_or_not:
                    valid_df_left.loc[index, 'match'] = 1

        # 正规则 软匹配
        valid_df_left.loc[:, 'soft_match_score'] = 0
        pos_rule_list = pos_rule_df['rule'].tolist()
        score_list = []
        if len(pos_rule_list) == 0:
            pass
        else:
            for index, row in valid_df_left[valid_df_left['match'] == 0].iterrows():
                sent_word_seq = row['processed_sent'].split()
                soft_match_score_sum = 0
                for rule in pos_rule_list:
                    rule_word_seq = rule.split('/')
                    soft_match_score = encoder.wv.n_similarity(sent_word_seq, rule_word_seq)
                    soft_match_score_sum = soft_match_score_sum + soft_match_score
                soft_match_score_aver = soft_match_score_sum / len(pos_rule_list) # 求平均数
                score_list.append(soft_match_score_aver)
            valid_df_left.loc[:, 'soft_match_score'] = score_list
            valid_df_left.loc[valid_df_left['soft_match_score'] >= threshold, 'match'] = 1

        TP = valid_df_left[(valid_df_left['label'] == 1) & (valid_df_left['match'] == 1)].shape[0]
        TN = valid_df_left[(valid_df_left['label'] == -1) & (valid_df_left['match'] == 0)].shape[0]
        FP = valid_df_left[(valid_df_left['label'] == 1) & (valid_df_left['match'] == 0)].shape[0]
        FN = valid_df_left[(valid_df_left['label'] == -1) & (valid_df_left['match'] == 1)].shape[0]

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if TP + FP == 0:
            precise = 0
        else:
            precise = TP / (TP + FP)
        if recall + precise == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precise / (recall + precise)
        with open(relation_filepath + "validation/experimental_reuslts/match_result.csv", 'a+') as f:
            f.writelines(
                str(iter_num)+ ',' + str(pos_rule_num) + ',' + str(neg_rule_num) + ',' + str(round(TP, 4)) + ',' + str(round(TN, 4)) + ',' + str(round(FP, 4)) + ',' + str(round(FN, 4)) + ',' + str(round(recall, 4)) + ',' + str(round(precise, 4)) + ',' + str(round(f1, 4)) + '\n'
                # str(iter_num)+ ',' + str(pos_rule_num) + ',' + str(neg_rule_num) + ',' + str(TP) + ',' + str(TN) + ',' + str(FP) + ',' + str(FN) + ',' + str(recall) + ',' + str(precise) + ',' + str(f1) + '\n'
            )