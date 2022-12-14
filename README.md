# AI_project_2

# 首先运行 git clone https://github.com/kfq2002/AI_project_2.git 下载文件，文件名为 AI_project_2

# 文件夹下有如下文件：
checkpoints/mymodel.pth: 此文件为自己搭建模型的预训练文件，已训练150个epoch，val acc约为64%，基本已达收敛；
data/: data文件夹下是已经按train:val:test=8:1:1划分好的数据集，为了方便程序调用，里面每一类图片的字符串编号都被换为了从0到18的数字，其对应关系与原数据集的类别顺序相同；
dataset.py: 此文件创建Dataset类，用于创建数据集类适应pytorch框架；
model.py: 网络模型文件；
train.py: 训练模型。

# 环境说明：
python: 3.8.10
pytorch: 1.13.0
CUDA: 11.7
torchvision: 0.14.0

其他用到的包：tqdm, argparse, wandb直接使用pip命令安装即可

# 代码运行参数介绍:
首先介绍train.py文件中ArgumentParser的各个参数：

--device 默认为"cuda:0"，可设置为"cpu"或"cuda:X"
--model 模型名字，默认为"mymodel"我的模型，可选项：经过改编的"resnet18", "resnet152", "densenet201"
--lr 学习率，默认为0.001
--lr_update 是否采用学习率更新方法，加上这个参数代表“是”，不加默认为否
--lr_step_size 若采用学习率更新，则多少个epoch更新一次，默认为10
--lr_update_gamma 若采用学习率更新，每次更新后的学习率与更新前学习率之比
--epoch 想要运行的epoch次数
--wandb 是否使用wandb平台记录训练数据和结果
--save_model 是否保存训练好的模型，默认为false
--resume 是否加载预训练模型，默认为false，如果助教想测试mymodel的话建议加上--resume的参数使用预训练模型

# 代码运行示例：
首先需要将目录移至AI_project_2下，即cd ./AI_project_2，否则部分文件路径会出错；

其次直接运行train.py文件即可

例1：加载预训练模型，运行mymodel：
python train.py --device cuda:1 --lr_update --lr_step_size 100 --epoch 300 --resume

例2：运行resnet18 model：
python train.py --device cuda:1 --model resnet18 --lr_update --epoch 30 

# 注：运行resnet等模型的时候会加载联网的pretrain模型，因此需要联网运行