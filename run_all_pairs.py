import os


# domain_list = ["a", "w", "d"]
domain_list = ["A", "C", "P", "R"]
record_file_path = "best_result.txt"
record = open(record_file_path, "w")
record.close()

for src in domain_list:
    for tgt in domain_list:
        if src == tgt:
            continue
        command = "python train_2.py --src %s --tgt %s" % (src, tgt)
        print(command)
        os.system(command)
