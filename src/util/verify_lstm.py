import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import time
import torch
import os
import collections
from torchtext.vocab import vocab


def verifier(relation_filepath, iter_num):
    print("第 " + str(iter_num) + " 轮验证开始")
    start_time = time.process_time()

    training_set_df = pd.read_csv(os.path.join(relation_filepath, '1_sent_to_mine', str(iter_num) + '.csv'))
        
    hard_match_result = pd.read_csv(relation_filepath + "5_match_result/hard_match/" + str(iter_num) + ".csv", index_col='sent_id')
    soft_match_result = pd.read_csv(relation_filepath + "5_match_result/soft_match/" + str(iter_num) + ".csv", index_col='sent_id')
    hard_matched = hard_match_result[hard_match_result['hard_match'] != 0]
    soft_matched = soft_match_result[soft_match_result['soft_match'] != 0]

    verifier = torch.load(os.path.join("/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/lstm",'saved_model','couple.model'))

    hard_matched.loc[:, 'model'] = model_predict(hard_matched, verifier)
    soft_matched.loc[:, 'model'] = model_predict(soft_matched, verifier)

    result, conflict_labeled = filter(hard_matched, soft_matched)
    result.to_csv(relation_filepath + '6_verify_result/' + str(iter_num) + '.csv')
    pd.concat([training_set_df, conflict_labeled]).to_csv(relation_filepath + "model_training_set/" + str(iter_num + 1) + ".csv", columns=['entity1', 'entity2', 'sent', 'processed_sent', 'human'])
    
    end_time = time.process_time()
    print("第 " + str(iter_num) + " 轮验证结束，耗时 " + str(end_time - start_time) + " 秒")
    return result

def get_vocab(sent_list):
    tokenized_data = [[word for word in sent.split(' ') if word != ''] for sent in sent_list]
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return vocab(counter, min_freq=1)


def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列表"""
    device = torch.device('cpu')
    sentence = torch.tensor([vocab.get_stoi()[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 1 if label.item() == 1 else -1


def model_predict(sent_df:pd.DataFrame, net):
    y = []
    sent_list = sent_df['processed_sent'].tolist()
    wordseq_list = [[word for word in sent.split(' ') if word != ''] for sent in sent_list]
    vocab = get_vocab(sent_list)
    for wordseq in wordseq_list:
        y.append(predict_sentiment(net, vocab, wordseq))
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