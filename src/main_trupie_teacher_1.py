from util.BiRNN import BiRNN
from util.miner import mine_rule
from util.label.original_label import sent_label_rule_oritinal
from util.match.hard_matcher import hard_match
from util.verify_hard_singlesvm import verifier
from util.merge.merge_labeled_sent import merge_labeled_sent
from util.label.final_label import final_label
from util.rule_deduplication import rule_dedupli
import time
from caculate import caculate

    
if __name__ == '__main__':
    relation_filepath = "/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/relation_data/teacher/1/"
    topk = 0
    topk_step = 300
    minlen = 3
    maxlen = 6 
    threshold = 0.7


    for iter_num in range(1, 11):
        topk = iter_num * topk_step
        start_time = time.process_time()
        mine_rule(relation_filepath, iter_num, 3, 6, topk)
        if iter_num >= 2:
            to_label_rule_num = rule_dedupli(relation_filepath, iter_num)[0]
            if to_label_rule_num == 0:
                print("没有可挖掘的规则，停止迭代")
                break
        sent_label_rule_oritinal(relation_filepath, iter_num)
        hard_match_result = hard_match(relation_filepath, iter_num)
        if hard_match_result.shape[0] == 0:
            print("没有能匹配的结果，停止迭代")
            break
        verifier(relation_filepath, iter_num)
        merge_labeled_sent(relation_filepath, iter_num)
        final_label(relation_filepath, iter_num)
        end_time = time.process_time()
        print("----------- 第 " + str(iter_num) + " 轮挖掘用时 " + str(end_time - start_time) + " 秒 ----------- ")
    caculate(relation_filepath)