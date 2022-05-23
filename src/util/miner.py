import pandas as pd
from prefixspan import PrefixSpan
from timeit import default_timer as timer


def mine_rule(relation_filepath, iter_num ,minlen, maxlen, topk):
    start_time = timer()
    print("第 " + str(iter_num) + " 轮规则挖掘开始")
    to_mine_filepath = relation_filepath + '1_sent_to_mine/' + str(iter_num) + '.csv'
    to_mine_sents_df = pd.read_csv(to_mine_filepath)
    miner_db = [word_seq_list.split() for word_seq_list in to_mine_sents_df['processed_sent'].tolist()]
    ps_mining_result = ps_mine(miner_db, minlen, maxlen, topk)
    mining_result = format_mining_results(ps_mining_result)
    save_filepath = relation_filepath + '2_rule_to_label/' + str(iter_num) + '.csv'
    mining_result.to_csv(save_filepath, index_label='rule_id')
    end_time = timer()
    print("第 %d 轮规则挖掘结束，耗时： %.2f 秒，挖掘出 %d 条规则" % (iter_num, (end_time - start_time), mining_result.shape[0]))
    mininglog(iter_num, mining_result, relation_filepath)
    return mining_result


def ps_mine(db, minlen, maxlen, topk):
    ps = PrefixSpan(db)
    ps.minlen=minlen
    ps.maxlen=maxlen
    return ps.topk(topk)


def format_mining_results(mining_result):
    rule_str_list = []
    rule_freq_list = []
    for rule in mining_result:
        rule_str = ''
        if "人物一" in rule[1] and "人物二" in rule[1]: # 过滤条件，很关键
            if (rule[1].count("人物一") + rule[1].count("人物二")) == len(rule[1]):
                pass
            else:
                for w in rule[1]:
                    rule_str = rule_str + w + '/'
                rule_str_list.append(rule_str.rstrip('/'))
                rule_freq_list.append(rule[0])
    rule_str_df = pd.DataFrame(rule_str_list, columns=['rule'])
    rule_freq_df = pd.DataFrame(rule_freq_list, columns=['frequence'])
    result = pd.concat([rule_str_df, rule_freq_df], axis=1)
    result.loc[:, 'label'] = 0 # 置为 0 方便标注
    return result

def mininglog(iter_num, mining_result:pd.DataFrame, relationfilepath:str):
    rule_num = mining_result.shape[0]
    with open(relationfilepath + "log/mining_log.txt", "a+") as f:
        f.writelines("第 " + str(iter_num) + " 轮挖掘出 " + str(rule_num) + " 条规则" + '\n')