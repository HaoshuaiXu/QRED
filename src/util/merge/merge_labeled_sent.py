import pandas as pd
from timeit import default_timer as timer


def merge_labeled_sent(relation_filepath, iter_num):
    print("第 %d 轮合并标注句子开始" % iter_num)
    start_time = timer()
    this = pd.read_csv(relation_filepath + '6_verify_result/' + str(iter_num) + ".csv", index_col='sent_id')
    last = pd.read_csv(relation_filepath + '1_sent_to_mine/' + str(iter_num) + ".csv", index_col='sent_id')
    result = pd.concat([this, last])
    result.drop_duplicates('sent', inplace=True)
    result.to_csv(relation_filepath + "1_sent_to_mine/" + str(iter_num + 1) + ".csv", columns=['entity1', 'entity2' , 'sent', 'processed_sent', 'human'])
    end_time = timer()
    print("第 %d 轮合并标注句子结束，耗时 %.2f 秒" % (iter_num, (end_time - start_time)))