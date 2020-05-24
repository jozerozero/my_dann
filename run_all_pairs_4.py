import os

domain_list = ["A", "C", "P", "R"]
record_file_path = "best_result_4.txt"
record = open(record_file_path, "w")
record.close()

for src in domain_list:
    for tgt in domain_list:
        if src == tgt:
            continue
        command = "python train_2.py --src %s --tgt %s --record %s" \
                  % (src, tgt, record_file_path)
        print(command)
        os.system(command)
