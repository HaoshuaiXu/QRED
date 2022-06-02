import os
from caculate import caculate

root_path = "/home/XuHaoshuai/Project/IQR-Iterative-Quality-oriented-Rule-Discovery-for-Relation-Extraction/diff_label_data/"
relation = "couple"
label_prop = 0
label_prop_value = str(label_prop) + "/"
relation_filepath = os.path.join(root_path, relation, label_prop_value)

caculate(relation_filepath)