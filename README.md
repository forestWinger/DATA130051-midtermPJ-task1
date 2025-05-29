# DATA130051-midtermPJ
25springCV期中pj task1

## 数据准备
https://data.caltech.edu/records/mzrjq-6wc02 下载caltech-101数据集 并之于data文件夹内

## 训练和推理
新建dir saved_models用于保存训练结果
新建dir pretrained_mw用于保存用imagenet预训练的resnet18模型

configs内有现成的实验超参数设置，便于复现结果，也可以自定义config。
运行 python train.py --config './configs/urconfigname.yaml'进行训练
运行 python eval.py --config './configs/urconfigname.yaml' 进行推理
