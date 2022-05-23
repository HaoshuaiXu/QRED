from util.BiRNN import BiRNN
from util.miner import mine_rule
from util.label.original_label import sent_label_rule_oritinal
from util.match.hard_matcher import hard_match
from util.match.soft_matcher import soft_match
# from util.match.soft_matcher_bert import soft_match
# from util.verify_hard_soft_singlesvm import verifier
from util.verify import verifier
# from util.verify_hard_singlesvm import verifier
# from util.verify_rule_self import verifier
# from util.verify_lstm import verifier
# from util.verify_fasttext import verifier
from util.verify_bert import verifier
from util.merge.merge_labeled_sent import merge_labeled_sent
from util.label.final_label import final_label
# from util.label.final_label_fasttext import final_label
# from util.label.final_label_bert import final_label
from util.merge.merge_unmatched_sent import merge_unmatched_sent
from util.rule_deduplication import rule_dedupli
import time
from transformers import BertTokenizer, BertModel

    
if __name__ == '__main__':
    relation_filepath = "/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/relation_data/brother/1/"
    topk = 0
    topk_step = 300
    minlen = 3
    maxlen = 6 
    threshold = 0.7

    final_label(relation_filepath, 1)