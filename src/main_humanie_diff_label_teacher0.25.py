import os
from util.miner import mine_rule
from util.label.original_label import sent_label_rule_oritinal
from util.match.hard_matcher import hard_match
from util.match.soft_matcher import soft_match
# from util.verify import verifier
from util.verify_diff_label import verifier
from util.merge.merge_labeled_sent import merge_labeled_sent
# from util.merge.merge_unmatched_sent import merge_unmatched_sent
from util.merge.merge_unmatched_sent_diff_label import merge_unmatched_sent
from util.label.final_label import final_label
from util.rule_deduplication import rule_dedupli
from timeit import default_timer as timer
from caculate import caculate

    
if __name__ == '__main__':
    root_path = "/home/XuHaoshuai/Project/IQR-Iterative-Quality-oriented-Rule-Discovery-for-Relation-Extraction/diff_label_data/"
    relation = "teacher"
    label_prop = 0.25
    label_prop_value = str(label_prop) + "/"
    relation_filepath = os.path.join(root_path, relation, label_prop_value)
    topk = 0
    topk_step = 300
    minlen = 3
    maxlen = 6 
    threshold = 0.7
    

    try:
        for iter_num in range(1, 10):
            topk = iter_num * topk_step
            start_time = timer()
            mine_rule(relation_filepath, iter_num, 3, 6, topk)
            if iter_num >= 2:
                to_label_rule_num = rule_dedupli(relation_filepath, iter_num)[0]
                if to_label_rule_num == 0:
                    print("没有可挖掘的规则，停止迭代")
                    break
            sent_label_rule_oritinal(relation_filepath, iter_num)
            hard_match(relation_filepath, iter_num)
            soft_match(relation_filepath, iter_num, threshold=threshold)
            verifier(relation_filepath, iter_num, label_prop)
            merge_labeled_sent(relation_filepath, iter_num)
            merge_unmatched_sent(relation_filepath, iter_num, label_prop)
            final_label(relation_filepath, iter_num)
            end_time = timer()
            print("----------- 第 %s 轮挖掘用时 %.2f 秒 ----------- " % (str(iter_num), (end_time - start_time)))
    finally:    
        caculate(relation_filepath)
    