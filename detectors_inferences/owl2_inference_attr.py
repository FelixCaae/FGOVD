import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from transformers import Owlv2Processor, Owlv2ForObjectDetection
# import spacy  # 用于更好的属性提取

# 下载必要的NLTK数据
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')

class AttributeAwareOwlViT:
    """属性感知的OwlViT推理，支持细粒度增强"""
    
    def __init__(self, 
                 model_path: str, 
                 processor_path: str,
                 attribute_vocab: List[str] = None,
                 device: str = "cuda"):
        """
        初始化属性感知的OwlViT
        
        Args:
            model_path: OwlViT模型路径
            processor_path: 处理器路径
            attribute_vocab: 属性词汇列表，如['red', 'large', 'wooden', ...]
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # 加载模型和处理器
        self.processor = Owlv2Processor.from_pretrained(processor_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # 属性词汇表
        if attribute_vocab is None:
            # 默认属性词汇表（可扩展）
            self.attribute_vocab = [
                'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray',
                'large', 'small', 'big', 'little', 'tall', 'short', 'wide', 'narrow',
                'wooden', 'metal', 'plastic', 'glass', 'fabric', 'leather', 'stone',
                'circular', 'square', 'rectangular', 'triangular', 'round', 'oval',
                'shiny', 'matte', 'transparent', 'opaque', 'reflective',
                'striped', 'spotted', 'patterned', 'plain', 'textured',
                'old', 'new', 'modern', 'antique', 'vintage',
                'soft', 'hard', 'smooth', 'rough', 'sharp', 'blunt'
            ]
        else:
            self.attribute_vocab = attribute_vocab
        
        # 加载spacy用于更好的属性提取
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("未找到spacy模型，使用简单规则提取属性")
            self.nlp = None
        
        # 停用词
        # self.stop_words = set(stopwords.words('english'))
    
    def extract_attributes(self, text: str) -> List[str]:
        """
        从文本中提取属性词
        
        Args:
            text: 输入文本，如"a large red wooden table"
            
        Returns:
            List[str]: 提取的属性词列表，如['large', 'red', 'wooden']
        """
        # if self.nlp is not None:
        #     # 使用spacy提取
        #     doc = self.nlp(text)
        #     attributes = []
            
        #     for token in doc:
        #         # 提取形容词和名词
        #         if token.pos_ in ['ADJ', 'NOUN'] and token.text.lower() in self.attribute_vocab:
        #             if token.text.lower() not in self.stop_words:
        #                 attributes.append(token.text.lower())
            
        #     return attributes
        
        # else:
        # 使用简单规则提取
        words = word_tokenize(text.lower())
        attributes = []
        
        for word in words:
            if word in self.attribute_vocab and word not in self.stop_words:
                attributes.append(word)
        
        return attributes
    
    def encode_attribute_texts(self, attributes: List[str]) -> torch.Tensor:
        """
        编码属性文本为特征
        
        Args:
            attributes: 属性词列表
            
        Returns:
            torch.Tensor: 属性特征 [num_attributes, feature_dim]
        """
        if not attributes:
            return None
        
        # 使用模型编码属性文本
        with torch.no_grad():
            attribute_inputs = self.processor(
                text=attributes,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # 获取文本特征
            # 注意：需要从模型中提取文本编码器的特征
            text_outputs = self.model.owlvit.text_model(**attribute_inputs)
            attribute_features = text_outputs.last_hidden_state.mean(dim=1)  # [num_attrs, dim]
            
        return attribute_features
    
    def extract_visual_features(self, image: np.ndarray) -> torch.Tensor:
        """
        提取图像的视觉特征
        
        Args:
            image: 输入图像 [H, W, 3]
            
        Returns:
            torch.Tensor: 视觉特征 [num_patches, feature_dim]
        """
        with torch.no_grad():
            # 预处理图像
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # 获取视觉特征
            import pdb;pdb.set_trace()
            vision_outputs = self.model.owlv2.vision_model(**inputs)
            # 通常取最后一层的[CLS] token或平均池化
            visual_features = vision_outputs.last_hidden_state[:, 1:, :]  # 去除[CLS] token
            visual_features = visual_features.mean(dim=1)  # 全局平均池化
            
        return visual_features
    
    def attribute_aware_similarity(self, 
                                  visual_features: torch.Tensor,
                                  attribute_features: torch.Tensor,
                                  main_text_feature: torch.Tensor,
                                  method: str = "weighted") -> torch.Tensor:
        """
        计算属性感知的相似度
        
        Args:
            visual_features: 视觉特征 [batch_size, feature_dim]
            attribute_features: 属性特征 [num_attributes, feature_dim]
            main_text_feature: 主文本特征 [1, feature_dim]
            method: 融合方法，可选"weighted", "max", "mean"
            
        Returns:
            torch.Tensor: 增强后的相似度分数
        """
        batch_size = visual_features.shape[0]
        
        if attribute_features is None or len(attribute_features) == 0:
            # 没有属性，使用原始相似度
            similarity = F.cosine_similarity(
                visual_features.unsqueeze(1),  # [batch, 1, dim]
                main_text_feature.unsqueeze(0),  # [1, 1, dim]
                dim=-1
            ).squeeze(1)
            return similarity
        
        num_attributes = len(attribute_features)
        
        # 计算与每个属性的相似度
        # visual_features: [batch, dim]
        # attribute_features: [num_attrs, dim]
        attr_similarities = []
        for i in range(num_attributes):
            attr_sim = F.cosine_similarity(
                visual_features,
                attribute_features[i:i+1].expand(batch_size, -1),
                dim=-1
            )
            attr_similarities.append(attr_sim)
        
        attr_similarities = torch.stack(attr_similarities, dim=1)  # [batch, num_attrs]
        
        # 计算主文本相似度
        main_similarity = F.cosine_similarity(
            visual_features,
            main_text_feature.expand(batch_size, -1),
            dim=-1
        )  # [batch]
        
        # 融合策略
        if method == "weighted":
            # 基于属性相似度加权的融合
            attr_weights = F.softmax(attr_similarities.mean(dim=0, keepdim=True), dim=1)  # [1, num_attrs]
            weighted_attr_sim = (attr_similarities * attr_weights).sum(dim=1)  # [batch]
            
            # 主相似度和属性相似度的加权平均
            alpha = 0.7  # 主相似度权重
            enhanced_similarity = alpha * main_similarity + (1-alpha) * weighted_attr_sim
            
        elif method == "max":
            # 取最大值
            max_attr_sim = attr_similarities.max(dim=1)[0]  # [batch]
            enhanced_similarity = torch.max(main_similarity, max_attr_sim)
            
        elif method == "mean":
            # 平均值
            mean_attr_sim = attr_similarities.mean(dim=1)  # [batch]
            enhanced_similarity = 0.5 * main_similarity + 0.5 * mean_attr_sim
            
        elif method == "attention":
            # 注意力机制融合
            # 计算注意力权重
            all_features = torch.cat([
                main_text_feature.unsqueeze(0),  # [1, dim]
                attribute_features  # [num_attrs, dim]
            ], dim=0)  # [1+num_attrs, dim]
            
            # 计算注意力
            attention_scores = torch.matmul(
                visual_features.unsqueeze(1),  # [batch, 1, dim]
                all_features.transpose(0, 1).unsqueeze(0)  # [1, dim, 1+num_attrs]
            ).squeeze(1)  # [batch, 1+num_attrs]
            
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, 1+num_attrs]
            
            # 加权相似度
            all_similarities = torch.cat([
                main_similarity.unsqueeze(1),  # [batch, 1]
                attr_similarities  # [batch, num_attrs]
            ], dim=1)  # [batch, 1+num_attrs]
            
            enhanced_similarity = (attention_weights * all_similarities).sum(dim=1)
        
        return enhanced_similarity
    
    def predict(self, 
                image: np.ndarray, 
                text_prompt: str,
                extract_attributes: bool = True,
                fusion_method: str = "weighted",
                nms_threshold: float = 0.5,
                max_predictions: int = 100) -> Dict:
        """
        属性感知的预测
        
        Args:
            image: 输入图像 [H, W, 3]
            text_prompt: 文本提示
            extract_attributes: 是否提取属性
            fusion_method: 融合方法
            nms_threshold: NMS阈值
            max_predictions: 最大预测数
            
        Returns:
            Dict: 预测结果
        """
        with torch.no_grad():
            # 1. 提取视觉特征
            visual_features = self.extract_visual_features(image)  # [1, dim]
            #这里有点要改的，应该提取密集特征，然后用正样本的框去做RoI Pooling得到Instance特征再去测分数
            # 2. 处理文本提示
            if extract_attributes:
                # 提取属性
                attributes = self.extract_attributes(text_prompt)
                print(f"提取的属性: {attributes}")
                
                # 编码属性
                attribute_features = self.encode_attribute_texts(attributes)
                
                # 编码主文本
                main_inputs = self.processor(
                    text=[text_prompt],
                    return_tensors="pt"
                ).to(self.device)
                main_text_outputs = self.model.owlvit.text_model(**main_inputs)
                main_text_feature = main_text_outputs.last_hidden_state.mean(dim=1)  # [1, dim]
                
                # 计算属性感知相似度
                similarity = self.attribute_aware_similarity(
                    visual_features,
                    attribute_features,
                    main_text_feature,
                    method=fusion_method
                )
                
            else:
                # 标准推理
                inputs = self.processor(
                    text=[text_prompt],
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                similarity = torch.sigmoid(outputs.logits[0]).max(dim=-1)[0]
            
            # 3. 获取检测结果
            inputs = self.processor(
                text=[text_prompt],
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # 处理输出
            logits = torch.max(outputs['logits'][0], dim=-1)
            scores = torch.sigmoid(logits.values).cpu().numpy()
            
            if extract_attributes:
                # 用属性增强的相似度替换原始分数
                scores = similarity.cpu().numpy()
            
            labels = logits.indices.cpu().numpy()
            boxes = outputs['pred_boxes'][0].cpu().numpy()
            
            # 4. 后处理
            height, width = image.shape[:2]
            converted_boxes = []
            for box in boxes:
                cx, cy, w, h = box
                x1 = (cx - w/2) * width
                y1 = (cy - h/2) * height
                x2 = (cx + w/2) * width
                y2 = (cy + h/2) * height
                converted_boxes.append([x1, y1, x2, y2])
            
            # 5. 过滤和排序
            data = list(zip(scores, converted_boxes, labels))
            sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
            
            # 6. NMS
            if nms_threshold > 0 and len(sorted_data) > 0:
                scores_nms = [x[0] for x in sorted_data]
                boxes_nms = [x[1] for x in sorted_data]
                labels_nms = [x[2] for x in sorted_data]
                
                from torchvision.ops import batched_nms
                
                boxes_tensor = torch.tensor(boxes_nms)
                scores_tensor = torch.tensor(scores_nms)
                labels_tensor = torch.tensor(labels_nms)
                
                keep_indices = batched_nms(boxes_tensor, scores_tensor, labels_tensor, nms_threshold)
                
                filtered_data = [sorted_data[i] for i in keep_indices]
                filtered_data = filtered_data[:max_predictions]
            else:
                filtered_data = sorted_data[:max_predictions]
            
            # 7. 格式化输出
            result = {
                'scores': [x[0] for x in filtered_data],
                'boxes': [x[1] for x in filtered_data],
                'labels': [x[2] for x in filtered_data],
                'text_prompt': text_prompt,
                'attributes_extracted': attributes if extract_attributes else []
            }
            
            return result

# 使用示例
def test_attribute_aware_owlvit():
    """测试属性感知的OwlViT"""
    import cv2
    
    # 初始化
    model_path ="epoch3_baseline/" # "google/owlv2-base-patch16-ensemble"
    processor_path = "/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlv2-base-patch16" #google/owlv2-base-patch16-ensemble"
    # processor = Owlv2Processor.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlv2-base-patch16")
    # model = Owlv2ForObjectDetection.from_pretrained("epoch3_baseline/")
    
    detector = AttributeAwareOwlViT(
        model_path=model_path,
        processor_path=processor_path,
        device="cuda"
    )
    
    # 加载测试图像
    image = cv2.imread("test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 测试提示
    test_prompts = [
        "a large red wooden table",  # 包含多个属性
        "a small black cat",         # 包含属性
        "a black tv",
        "a red tv",
        "a blue tv",
        "a person",                  # 不包含明显属性
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"测试提示: {prompt}")
        print('='*50)
        
        # 标准推理
        print("\n1. 标准推理:")
        result_standard = detector.predict(
            image, prompt, extract_attributes=False
        )
        print(f"   检测到 {len(result_standard['boxes'])} 个物体")
        if len(result_standard['scores']) > 0:
            print(f"   最高分数: {max(result_standard['scores']):.4f}")
        
        # 属性感知推理
        print("\n2. 属性感知推理:")
        result_enhanced = detector.predict(
            image, prompt, extract_attributes=True, fusion_method="weighted"
        )
        print(f"   检测到 {len(result_enhanced['boxes'])} 个物体")
        if len(result_enhanced['scores']) > 0:
            print(f"   最高分数: {max(result_enhanced['scores']):.4f}")
        
        # 比较
        if len(result_standard['scores']) > 0 and len(result_enhanced['scores']) > 0:
            standard_max = max(result_standard['scores'])
            enhanced_max = max(result_enhanced['scores'])
            improvement = (enhanced_max - standard_max) / standard_max * 100
            print(f"\n   分数提升: {improvement:.1f}%")

# 批量测试函数
def evaluate_attribute_awareness(dataset_path: str, 
                                detector: AttributeAwareOwlViT,
                                num_samples: int = 100):
    """
    在数据集上评估属性感知策略的效果
    
    Args:
        dataset_path: 数据集路径
        detector: 属性感知检测器
        num_samples: 测试样本数
    """
    import json
    from tqdm import tqdm
    
    # 加载数据集
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    improvements = []
    attribute_counts = []
    
    # 随机采样
    import random
    samples = random.sample(data['annotations'], min(num_samples, len(data['annotations'])))
    
    for ann in tqdm(samples, desc="评估属性感知"):
        # 获取图像
        image_id = ann['image_id']
        image_info = next(img for img in data['images'] if img['id'] == image_id)
        image_path = image_info['file_name']
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 构建提示
        category_id = ann['category_id']
        category = next(cat for cat in data['categories'] if cat['id'] == category_id)
        prompt = f"a {category['name']}"
        
        # 添加属性（如果有）
        if 'attributes' in ann:
            for attr in ann['attributes']:
                prompt += f" {attr}"
        
        # 标准推理
        result_std = detector.predict(image, prompt, extract_attributes=False)
        
        # 属性感知推理
        result_enh = detector.predict(image, prompt, extract_attributes=True)
        
        # 记录属性数量
        attributes = detector.extract_attributes(prompt)
        attribute_counts.append(len(attributes))
        
        # 比较分数
        if result_std['scores'] and result_enh['scores']:
            std_score = max(result_std['scores'])
            enh_score = max(result_enh['scores'])
            
            if std_score > 0:
                improvement = (enh_score - std_score) / std_score * 100
                improvements.append(improvement)
    
    # 分析结果
    if improvements:
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        print(f"\n评估结果:")
        print(f"  平均提升: {avg_improvement:.2f}%")
        print(f"  标准差: {std_improvement:.2f}%")
        print(f"  最大提升: {max(improvements):.2f}%")
        print(f"  最小提升: {min(improvements):.2f}%")
        
        # 按属性数量分组
        import pandas as pd
        df = pd.DataFrame({
            'improvement': improvements[:len(attribute_counts)],
            'num_attributes': attribute_counts
        })
        
        group_stats = df.groupby('num_attributes')['improvement'].agg(['mean', 'count'])
        print(f"\n按属性数量的提升:")
        print(group_stats)
    
    return improvements, attribute_counts

if __name__ == "__main__":
    # 运行测试
    test_attribute_aware_owlvit()