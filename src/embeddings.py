import torch
import torch.nn as nn

torch.manual_seed(42)

vocab_size = 30   # 词表大小：假设词表一共有30个Token 0~29
embedding_dim = 8  # 每个 Token 用 8 维向量表示（GPT-2 是 768）

# 创建Embedding层
# 一个形状为 (30, 8) 的可训练参数矩阵，初始值是随机的
# 可训练的查找表，把每个Token ID映射成一个高维向量
'''
这张表会在后续的训练过程中，通过反向传播不断更新，直到里面的数值真正承载起语义信息
不同任务、不同训练数据训练出来的Embedding层，即便词表相同，里面的向量数值也完全不同 
'''
# 但是 Embedding层不知道 语序
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# after tokenizer
token_ids = torch.tensor([3, 15, 20])
# 查表操作：输入Token ID，返回对应的向量
token_vectors = embedding_layer(token_ids)

print(token_vectors.shape) # torch.Size([3, 8]): 3个Token，每个Token被表示成一个8维向量
print(token_vectors)
# tensor([[ 1.2791,  1.2964,  0.6105,  1.3347, -0.2316,  0.0418, -0.2516,  0.8599],
#         [-0.0127,  0.2408,  0.1325,  0.7642,  1.0950,  0.3399,  0.7200,  0.4114],
#         [-1.9006,  0.2286,  0.0249, -0.3460,  0.2868, -0.7308,  0.1748, -1.0939]],
#        grad_fn=<EmbeddingBackward0>)

print(embedding_layer.weight[token_ids])


'''
Positional Encoding: 给向量加上「位置」信息
Transformer论文的解法：用一组不同频率的正弦 / 余弦函数来编码位置
'''

'''
Self-Attention 不看顺序，顺序 必须 人手动"喂"给模型
位置编码的公式为正余弦函数： [每一对维度对应一个固定频率的波，i 越大波长越长、变化越"慢"]
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
选择这个函数是因为对任意固定偏移量 k，PE_pos+k 都可以表示成 PE_pos 的一个线性函数
"正余弦"这个具体形式不是唯一解，可学习的位置向量也能达到类似效果; 但可学习的位置向量只认识训练时见过的那些位置；
固定公式的正余弦不管多长的位置都能直接算出来，泛化性更好
'''
import math
