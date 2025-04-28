#!/bin/bash
#SBATCH --job-name=angle_pred     # 作业名称
#SBATCH --output=./log/output_%j.txt   # 标准输出文件名（%j将被替换为作业ID）
#SBATCH --error=./log/error_%j.txt     # 标准错误文件名
#SBATCH --nodelist=node06       # 指定运行节点
#SBATCH --ntasks=1              # 请求一个任务
#SBATCH --cpus-per-task=8       # 每个任务使用的CPU核心数
##SBATCH --gres=gpu:1             # 请求一个GPU
#SBATCH --partition=gpu          # 分区名称（partition）
#SBATCH --mem=64G                # 内存请求


# 打印一些作业信息
echo "作业开始时间: `date`"
echo "作业运行在节点: `hostname`"
echo "作业分配的ID: $SLURM_JOB_ID"

# 加载Conda环境
source ~/.bashrc
conda activate torch310

# 确保在正确的目录
cd /public/home/yz/python_project/yolov5_drone

# 运行Python脚本
echo "开始运行Python脚本..."
python train.py  --data ./data/mydata.yaml --epoch 100 --batch-size 16  --single-cls

echo "作业结束时间: `date`"

