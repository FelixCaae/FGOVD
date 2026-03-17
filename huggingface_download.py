import os
# // 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# // 下载模型
os.system(r'huggingface-cli download --resume-download google/owlv2-base-patch16 --local-dir /home/wuke_2024/ov202503/cz_github/FGOVD/weights/google/owlv2-base-patch16')
# os.system(r'huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir /home/wuke_2024/ov202503/text_encoder/bert-base-uncased')
# // 下载数据集
# os.system('huggingface-cli download --repo-type dataset --resume-download HF上的数据集名称 --local-dir 本地存放路径')
