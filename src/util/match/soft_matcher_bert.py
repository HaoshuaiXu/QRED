import pandas as pd
from transformers import BertTokenizer, BertModel
import time
import torch


def soft_match(relation_filepath, iter_num, threshold, tokenizer:BertTokenizer, model:BertModel):
    print("第 " + str(iter_num) + " 轮软匹配开始")
    start_time = time.process_time()
    rule_df = pd.read_csv(relation_filepath + "3_original_label_rule/" + str(iter_num) + ".csv", index_col='rule_id')
    sent_df = pd.read_csv(relation_filepath + "4_sent_to_match/" + str(iter_num) + ".csv", index_col='sent_id')
    match_result = match(rule_df, sent_df, tokenizer, model, threshold)
    match_result.to_csv(relation_filepath + "5_match_result/soft_match/" + str(iter_num) + ".csv")
    end_time = time.process_time()
    print("第 " + str(iter_num) + " 轮软匹配结束，耗时 " + str(end_time - start_time) + " 秒")
    print("软匹配数量 " + str(match_result.shape[0]) + " 条")
    print("其中匹配上的正例 " + str(match_result[match_result['soft_match'] == 1].shape[0]) + " 匹配上的负例 " + str(match_result[match_result['soft_match'] == -1].shape[0]))
    soft_match_log(match_result, iter_num, relation_filepath)
    return match_result


def get_embedding(text:str, tokenizer:BertTokenizer, model:BertModel):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    sent_vec = torch.mean(last_hidden_states[0], dim=0)
    return sent_vec


def match(rule_df:pd.DataFrame, sent_df:pd.DataFrame, tokenizer, model, threshold=0.7):
    sent_df['soft_match_score'] = 0
    sent_df['soft_match'] = 0
    neg_rule_list = rule_df[rule_df['original_label'] == -1]['rule'].tolist()
    pos_rule_list = rule_df[rule_df['original_label'] == 1]['rule'].tolist()
    score_list = []
    if len(neg_rule_list) == 0:
        pass
    else:
        for index, row in sent_df.iterrows():
            soft_match_score_sum = 0
            for rule in neg_rule_list:
                str = ''
                rule = str.join(rule.split('/'))
                rule_vec = get_embedding(rule, tokenizer, model)
                sent_vec = get_embedding(row['processed_sent'], tokenizer, model)
                soft_match_score = torch.cosine_similarity(rule_vec, sent_vec, dim=0).item()
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
                str = ''
                rule = str.join(rule.split('/'))
                rule_vec = get_embedding(rule)
                sent_vec = get_embedding(row['processed_sent'])
                soft_match_score = torch.cosine_similarity(rule_vec, sent_vec, dim=0).item()
                soft_match_score_sum = soft_match_score_sum + soft_match_score
            soft_match_score_aver = soft_match_score_sum / len(pos_rule_list) # 求平均数
            score_list.append(soft_match_score_aver)
        sent_df.loc[sent_df['soft_match'] == 0, 'soft_match_score'] = score_list
        sent_df.loc[(sent_df['soft_match_score'] >= threshold) & (sent_df['soft_match'] == 0), 'soft_match'] = 1

        return sent_df


def soft_match_log(match_result:pd.DataFrame, iter_num, relationfilepath):
    pos_match_num = match_result[match_result['soft_match'] == 1].shape[0]
    neg_match_num = match_result[match_result['soft_match'] == -1].shape[0]
    no_match_num = match_result[match_result['soft_match'] == 0].shape[0]
    with open(relationfilepath + 'log/soft_match_log.txt', 'w+') as f:
        f.writelines(str(iter_num) + ',' + str(pos_match_num) + ',' + str(neg_match_num) + ',' + str(no_match_num) + '\n')