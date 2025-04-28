# 模型路径
weights_path=./runs/train/exp1/weights/best.pt

# 测试集验证
python val.py   --weights $weights_path \
                --data data/mydata.yaml \
                --task test \
                --project runs/val_with_error_analysis \
                --name exp1


# 测试集画框
python detect.py --weights $weights_path