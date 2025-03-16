import os
weights_folder = "/home/nenu/a25/lsh/1193ssssss/mytrain/models"


for i in range(16,-1,-1):
    weight_file = os.path.join(weights_folder,f"weights_{i:01d}")
    if not os.path.exists(weight_file):
        print(f"weight file {weight_file} not found")
        continue

    # 评估命令
    evaluation_command = f"python /home/nenu/a25/lsh/seg-mono/hr-mono/evaluate_depth.py --load_weights_folder {weight_file} --data_path /home/nenu/a25/lsh/z-litti --model MG-Mono"

    # 执行评估命令
    print(f"evaluating weights {i:01d}")
    os.system(evaluation_command)
