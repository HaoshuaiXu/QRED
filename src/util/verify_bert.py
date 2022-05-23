import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn import svm
import time


def verifier(relation_filepath, iter_num, tokenizer, model):
    print("第 " + str(iter_num) + " 轮验证开始")
    start_time = time.process_time()
    training_set_df = pd.read_csv(relation_filepath + "model_training_set/" + str(iter_num) + ".csv", index_col='sent_id')
    verify_model = pre_train(training_set_df, tokenizer, model)
    
    hard_match_result = pd.read_csv(relation_filepath + "5_match_result/hard_match/" + str(iter_num) + ".csv", index_col='sent_id')
    soft_match_result = pd.read_csv(relation_filepath + "5_match_result/soft_match/" + str(iter_num) + ".csv", index_col='sent_id')
    hard_matched = hard_match_result[hard_match_result['hard_match'] != 0]
    soft_matched = soft_match_result[soft_match_result['soft_match'] != 0]

    hard_matched.loc[:, 'model'] = model_predict(hard_matched, verify_model, tokenizer, model)
    soft_matched.loc[:, 'model'] = model_predict(soft_matched, verify_model, tokenizer, model)

    result, conflict_labeled = filter(hard_matched, soft_matched)
    result.to_csv(relation_filepath + '6_verify_result/' + str(iter_num) + '.csv')
    pd.concat([training_set_df, conflict_labeled]).to_csv(relation_filepath + "model_training_set/" + str(iter_num + 1) + ".csv", columns=['entity1', 'entity2', 'sent', 'processed_sent', 'human'])
    
    end_time = time.process_time()
    print("第 " + str(iter_num) + " 轮验证结束，耗时 " + str(end_time - start_time) + " 秒")
    return result


def pre_train(sent_df:pd.DataFrame, tokenizer, model):
    X_train = sent2vec(sent_df['processed_sent'], tokenizer, model)
    y_train = sent_df['human'].tolist()
    verify_model = svm.SVC()
    verify_model.fit(X_train, y_train)
    return verify_model


def get_embedding(text:str, tokenizer:BertTokenizer, model:BertModel):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    sent_vec = torch.mean(last_hidden_states[0], dim=0)
    return sent_vec

def sent2vec(sent_list, tokenizer, model):
    X = []
    for sent in sent_list:
        X.append(get_embedding(sent, tokenizer, model))
    return X


def model_predict(sent_df:pd.DataFrame, verify_model:svm.SVC, tokenizer, model):
    X_pred = sent2vec(sent_df, tokenizer, model)
    y = verify_model.predict(X_pred)
    return y


def filter(hard_df:pd.DataFrame, soft_df:pd.DataFrame):
    hard_match_same = hard_df[hard_df['hard_match'] == hard_df['model']]
    hard_match_conflict = hard_df[hard_df['hard_match'] != hard_df['model']]

    soft_match_same = soft_df[soft_df['soft_match'] == soft_df['model']]
    soft_match_conflict = soft_df[soft_df['soft_match'] != soft_df['model']]

    hard_match_same.loc[:, 'human'] = hard_match_same['model']
    soft_match_same.loc[:, 'human'] = soft_match_same['model']

    same_merge = pd.concat([
        pd.DataFrame(hard_match_same, columns=['sent_id','entity1', 'entity2', 'sent', 'processed_sent', 'human']),
        pd.DataFrame(soft_match_same, columns=['sent_id', 'entity1', 'entity2', 'sent', 'processed_sent', 'human'])
    ])

    hard_match_conflict.loc[:, 'human'] = hard_match_conflict['label']
    soft_match_conflict.loc[:, 'human'] = soft_match_conflict['label']

    conflict_merge = pd.concat(
        [
            pd.DataFrame(hard_match_conflict, columns=['sent_id', 'entity1', 'entity2', 'sent', 'processed_sent', 'human']),
            pd.DataFrame(soft_match_conflict, columns=['sent_id', 'entity1', 'entity2', 'sent', 'processed_sent', 'human'])
        ]
    )

    filter_output = pd.concat([same_merge, conflict_merge])
    return filter_output, conflict_merge