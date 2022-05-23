import pandas as pd


def mininglog(iter_num, mining_result:pd.DataFrame, relationfilepath:str):
    rule_num = mining_result.shape[0]
    with open(relationfilepath + "log/mining_log.txt", "a+") as f:
        f.writelines("第 " + str(iter_num) + " 轮挖掘出 " + str(rule_num) + " 条规则")