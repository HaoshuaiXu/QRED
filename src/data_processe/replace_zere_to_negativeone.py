import pandas as pd



relation_filepath = "/Users/xuhaoshuai/Project/HumanIE-IPM-experiment-2.0/relation_data/parentschild/"

seed = pd.read_csv(relation_filepath + "all_data_deduplication.csv", index_col='sent_id')
print(seed[seed['label'] == 0].shape)
seed.loc[seed['label'] == 0, 'label'] = -1
print(seed[seed['label'] == -1].shape)
seed.to_csv(relation_filepath + "all_data_deduplication.csv")