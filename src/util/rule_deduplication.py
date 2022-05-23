import pandas as pd


def rule_dedupli(relation_filepath, iter_num):
    to_label = pd.read_csv(relation_filepath + "2_rule_to_label/" + str(iter_num) + ".csv")
    original_num = to_label.shape[0]
    labeled = pd.read_csv(relation_filepath + "rule_set/"  + str(iter_num - 1) + ".csv")
    for index, row in to_label.iterrows():
        if row['rule'] in labeled['rule'].tolist():
            to_label.drop(index=index, inplace=True)
    now_num = to_label.shape[0]
    to_label.to_csv(relation_filepath + "2_rule_to_label/"  + str(iter_num) + ".csv")
    print("第 %d 轮删除了 %d 条规则 " % (iter_num, (original_num - now_num)))
    return to_label.shape