import pandas as pd
from prefixspan import PrefixSpan


class RuleMining:
    def __init__(self, sent_df:pd.DataFrame, processed_sent_label:str) -> None:
        self.sent_df = sent_df
        self.mining_db
        self.mining_result
        self.formatted_mining_result
    
    def construct_mining_db(self, processed_sent_list:list):
        self.mining_db = [word_seq_list.split() for word_seq_list in processed_sent_list]
    
    def get_mining_db(self):
        return self.mining_db
    
    def mine_rules(self, minlen, maxlen, topk):
        ps = PrefixSpan(self.mining_db)
        ps.minlen=minlen
        ps.maxlen=maxlen
        self.mining_result = ps.topk(topk)
    
    def get_mining_result(self):
        return self.mining_result
    
    def format_mining_result(self):
        rule_str_list = []
        rule_freq_list = []
        for rule in self.mining_result:
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
        self.formatted_mining_result = result



def mine_rule(sent_df, saved_filepath, minlen=3, maxlen=6, topk=200):
    miner_db = [word_seq_list.split() for word_seq_list in sent_df['processed_sent'].tolist()]
    ps_mining_result = ps_mine(miner_db, minlen, maxlen, topk)
    mining_result = format_mining_results(ps_mining_result)
    mining_result = rule2pattern(mining_result)
    mining_result.to_csv(saved_filepath, index_label='rule_id')
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
    result['label'] = 0 # 置为 0 方便标注
    return result

def rule2pattern(ruledf:pd.DataFrame):
    patlist = []
    for rule in ruledf['rule'].tolist():
        pat = ''
        for word in rule.split('/'):
            pat = pat + '.*' + word
        patlist.append(pat + '.*')
    ruledf['pattern'] = patlist
    return ruledf