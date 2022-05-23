from os import remove
import pandas as pd


def remove_duplicate(df_filepath, index_col_name, duplicate_label_name, output_name):
    df = pd.read_csv(df_filepath, index_col=index_col_name)
    print(df.shape)
    df = df.drop_duplicates(subset=[duplicate_label_name])
    print(df.shape)
    df.to_csv(output_name)

if __name__ == '__main__':
    relation_filepath = "/Users/xuhaoshuai/Project/HumanIE-IPM-experiment-2.0/relation_data/parentschild/"
    remove_duplicate(
        df_filepath=relation_filepath + "all_data.csv",
        index_col_name='sent_id',
        duplicate_label_name='sent',
        output_name=relation_filepath + "all_data_deduplication.csv"
    )