from glob import glob
import os
import shutil

for data_path in glob(os.path.join("*.txt")):
    print(data_path)
    data_list = list()
    for line in open(data_path).readlines():
        content = line.strip().replace("/data", "dataset")
        data_list.append(content)
    new_data_file = open(data_path, mode="w")
    for content in data_list:
        print(content, file=new_data_file)
    # exit()
