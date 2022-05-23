import pandas as pd
from gensim.models import FastText
import re
import time


def final_label(relation_filepath, iter_num):
    print("第 " + str(iter_num) + " 轮规则最终标签标注开始")
    start_time = time.process_time()
    rule_df = pd.read_csv(relation_filepath + "3_original_label_rule/" + str(iter_num) + ".csv")
    sent_df = pd.read_csv(relation_filepath + "1_sent_to_mine/" + str(iter_num + 1) + ".csv", index_col='sent_id')
    rule_df['pos_sent_num'] = 0
    rule_df['neg_sent_num'] = 0
    result = filter(do_label(sent_df, rule_df))
    if iter_num == 1:
        result.to_csv(relation_filepath + "7_final_label_rule/" + str(iter_num) + ".csv", index=None)
        result.to_csv(relation_filepath + "rule_set/" + str(iter_num) + ".csv", index=None)
    else:
        result.to_csv(relation_filepath + "7_final_label_rule/" + str(iter_num) + ".csv", index=None)
        result = result[['rule', 'pattern', 'frequence', 'final_label']]
        last_rule_set = pd.read_csv(relation_filepath + "rule_set/" + str(iter_num - 1) + ".csv")
        pd.concat([last_rule_set, result]).to_csv(relation_filepath + "rule_set/" + str(iter_num) + ".csv", index=None)
    end_time = time.process_time()
    print("第 " + str(iter_num) + " 轮规则最终标签标注结束，耗时 " + str(end_time - start_time) + " 秒")


def do_label(sent_df:pd.DataFrame, rule_df:pd.DataFrame):
    encoder = FastText.load("/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/fasttext/fasttext.model")
    for sent_index, sent_row in sent_df.iterrows():
        for rule_index, rule_row in rule_df.iterrows():
            match_or_not = re.search(pattern=rule_row['pattern'], string=sent_row['processed_sent'])
            if match_or_not:
                if sent_row['human'] == 1:
                    rule_df.loc[rule_index, 'pos_sent_num'] = rule_df.loc[rule_index, 'pos_sent_num'] + 1
                if sent_row['human'] == -1:
                    rule_df.loc[rule_index, 'neg_sent_num'] = rule_df.loc[rule_index, 'neg_sent_num'] + 1
            else:
                soft_match_score = encoder.wv.n_similarity(sent_row['processed_sent'].split(), rule_row['rule'].split('/'))
                if soft_match_score >= 0.7:
                    if sent_row['human'] == 1:
                        rule_df.loc[rule_index, 'pos_sent_num'] = rule_df.loc[rule_index, 'pos_sent_num'] + 1
                    if sent_row['human'] == -1:
                        rule_df.loc[rule_index, 'neg_sent_num'] = rule_df.loc[rule_index, 'neg_sent_num'] + 1
                else:
                    pass
    return rule_df


def filter(rule_df:pd.DataFrame):
    rule_df['final_label'] = 0
    rule_df.loc[rule_df['pos_sent_num'] > rule_df['neg_sent_num'], 'final_label'] = 1
    rule_df.loc[rule_df['pos_sent_num'] < rule_df['neg_sent_num'], 'final_label'] = -1
    return rule_df

