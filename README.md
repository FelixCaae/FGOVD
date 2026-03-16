## 配置
1. 下载owlv2-base-patch16权重到weights文件夹中
2. pip install -r requirements.txt
## 训练，多卡
python main_ddp.py --world-size 8 --config configs/basev2.yaml
