## 配置
1. 下载owlv2-base-patch16权重到weights文件夹中
2. 下载lvis_v1_val.json (lvis验证集标签)
3. 下载FG-OVD的json文件，https://lorebianchi98.github.io/FG-OVD，修改basev2.yaml
4. pip install -r requirements.txt
## 训练，多卡
python main_ddp.py --world-size 8 --config configs/basev2.yaml --amp #使用--amp开启混合精度

