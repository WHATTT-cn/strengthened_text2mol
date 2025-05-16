import torch
from transformers import BertModel, BertTokenizerFast

# 下载并保存模型
model_name = 'allenai/scibert_scivocab_uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# 保存到指定目录
save_path = './pretrained_models/scibert'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
