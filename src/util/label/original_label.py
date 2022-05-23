import pandas as pd
import re
from timeit import default_timer as timer


def sent_label_rule_oritinal(relation_filepath, iter_num):
    print("第 " + str(iter_num) + " 轮规则标注开始")
    start_time = timer()
    ruledf = pd.read_csv(relation_filepath + '2_rule_to_label/' + str(iter_num) + ".csv", index_col='rule_id')
    sentdf = pd.read_csv(relation_filepath + '1_sent_to_mine/' + str(iter_num) + ".csv", index_col='sent_id')
    ruledf.loc[:, 'pos_sent_num'] = 0
    ruledf.loc[:, 'neg_sent_num'] = 0
    ruledf = decide_label(match_to_label(sentdf, rule2pattern(ruledf)))
    ruledf.rename(
        columns={'label': 'original_label'})\
            .to_csv(
                relation_filepath + "3_original_label_rule/" + str(iter_num) + ".csv",
                columns=['rule', 'pattern', 'frequence', 'original_label'])
    end_time = timer()
    # print("第 " + str(iter_num) + " 轮规则标注结束，耗时 " + str(end_time - start_time) + " 秒")
    print("第 %d 轮规则标注结束，正规则 %d 条，负规则 %d 条，耗时 %.2f 秒" \
        % (iter_num, ruledf[ruledf['label'] == 1].shape[0], ruledf[ruledf['label'] == -1].shape[0] ,(end_time - start_time)))
    return ruledf


def rule2pattern(ruledf:pd.DataFrame):
    patlist = []
    for rule in ruledf['rule'].tolist():
        pat = ''
        for word in rule.split('/'):
            pat = pat + '.*' + word
        patlist.append(pat + '.*')
    ruledf.loc[:, 'pattern'] = patlist
    return ruledf


def match_to_label(sent_df, rule_df):
    for sent_index, sent_row in sent_df.iterrows():
        for rule_index, rule_row in rule_df.iterrows():
            match_or_not = re.search(pattern=rule_row['pattern'], string=sent_row['processed_sent'])
            if match_or_not:
                if sent_row['human'] == 1:
                    rule_df.at[rule_index, 'pos_sent_num'] = rule_df.at[rule_index, 'pos_sent_num'] + 1
                if sent_row['human'] == -1:
                    rule_df.at[rule_index, 'neg_sent_num'] = rule_df.at[rule_index, 'neg_sent_num'] + 1
    return rule_df


def decide_label(rule_df:pd.DataFrame):
    rule_df.loc[rule_df['pos_sent_num'] / rule_df['neg_sent_num'] >= 2.3333, 'label'] = 1
    rule_df.loc[rule_df['pos_sent_num'] / rule_df['neg_sent_num'] <= 0.4286, 'label'] = -1
    rule_df.loc[(rule_df['pos_sent_num'] / rule_df['neg_sent_num'] < 2.3333) & (rule_df['pos_sent_num'] / rule_df['neg_sent_num'] > 0.4286), 'label'] = 0
    # rule_df.loc[rule_df['pos_sent_num'] > rule_df['neg_sent_num'], 'label'] = 1
    # rule_df.loc[rule_df['pos_sent_num'] < rule_df['neg_sent_num'], 'label'] = -1
    # rule_df.loc[rule_df['pos_sent_num'] == rule_df['neg_sent_num'], 'label'] = 0
    return rule_df


def original_label_log(ruledf:pd.DataFrame, relationfilepath:str, iter_num):
    pos_num = ruledf[ruledf['label'] == 1].shape[0]
    neg_num = ruledf[ruledf['label'] == -1].shape[0]
    with open(relationfilepath + "log/original_label_log.txt", "w+") as f:
        f.writelines("第 " + str(iter_num) + " 轮正规则有 " + str(pos_num) + " 负规则有 " + str(neg_num))

    