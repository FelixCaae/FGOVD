## 配置
1. 下载owlv2-base-patch16权重到weights文件夹中
2. 下载lvis_v1_val.json (lvis验证集标签)
3. 下载FG-OVD的json文件，https://lorebianchi98.github.io/FG-OVD，修改basev2.yaml
4. pip install -r requirements.txt
## 训练，多卡
使用--amp 开启混合精度，但是有训崩了的风险
python main_ddp.py --world_size 8 --config configs/base-v2.yaml 

