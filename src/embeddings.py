import torch
import torch.nn as nn

torch.manual_seed(42)

vocab_size = 30   # 词表大小：假设词表一共有30个Token 0~29
embedding_dim = 8  # 每个 Token 用 8 维向量表示（GPT-2 是 768）

# 创建Embedding层
# 一个形状为 (30, 8) 的可训练参数矩阵，初始值是随机的
# 可训练的查找表，把每个Token ID映射成一个高维向量 (训练出来的参数，理论上可以是任意实数)
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
位置编码的公式为正余弦函数：  sin 和 cos 的输出永远严格落在 [-1, 1] 区间内 -- 解决"数值失控"问题
[每一对维度对应一个固定频率的波，i 越大波长越长、变化越"慢"]
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
[i 越大，波长越长，变化越"慢"；i 越小，波长越短，变化越"快"]
(这个组合几乎不会出现重复的"指纹")
选择这个函数是因为对任意固定偏移量 k，PE_pos+k 都可以表示成 PE_pos 的一个线性函数
"正余弦"这个具体形式不是唯一解，可学习的位置向量也能达到类似效果; 但可学习的位置向量只认识训练时见过的那些位置；
固定公式的正余弦不管多长的位置都能直接算出来，泛化性更好
'''
import math

vocab = {"cat":0, "is":1, "cute":2}  # 词表，把词映射成 Token ID
d_model = 8
max_len = 10   # 最多生成 10 个位置的编码 (>= 词表大小)

'''
为序列中不同位置的 Token 生成不同的位置向量，让 Transformer 知道每个词处于句子的什么位置
'''
def get_positional_embedding(max_len, d_model):
    pe = torch.zeros(max_len, d_model) # 每一行表示一个位置，每一列表示位置向量的一个维度
    print(pe.shape) # torch.Size([10, 8])

    # .unsqueeze(1) 在第 1 个维度增加一个维度，将一维向量变成列向量
    position = torch.arange(0, max_len).unsqueeze(1).float()  # 整数转换成浮点数
    print(position.shape) # torch.Size([10, 1])

    # 10000^(2i/d_model) 的倒数
    # 用 exp(log(...)) 是为数值计算更稳定，效果和算 10000^(2i/d_model) 一样
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )  # torch.arange(0, d_model, 2) -> 2i

    # 所有行，取第 0, 2, 4, 6 列
    pe[:,0::2] = torch.sin(position * div_term)
    # 所有行，取第 1, 3, 5, 7 列
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

pe = get_positional_embedding(max_len, d_model)
print(pe[:3])
# tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
#           1.0000e+00,  0.0000e+00,  1.0000e+00],
#         [ 8.4147e-01,  5.4030e-01,  9.9833e-02,  9.9500e-01,  9.9998e-03,
#           9.9995e-01,  1.0000e-03,  1.0000e+00],
#         [ 9.0930e-01, -4.1615e-01,  1.9867e-01,  9.8007e-01,  1.9999e-02,
#           9.9980e-01,  2.0000e-03,  1.0000e+00]])

embedding = nn.Embedding(len(vocab), d_model)

def encode(sentence_token):
    # 将 Token 转换成 ID
    ids = torch.tensor([vocab[t] for t in sentence_token])
    # 获取词向量
    emb = embedding(ids)
    '''
    位置编码的数量必须与句子中的 Token 数量一致
    自动适配不同长度的句子
    '''
    pos_enc = pe[: len(sentence_token)]
    # # 语义 + 位置，两者相加
    return emb + pos_enc

cat_is_cute = encode(["cat", "is", "cute"])
cute_is_cat = encode(["cute", "is", "cat"])
print(cat_is_cute)
# tensor([[-0.9291,  1.2762, -0.5389,  1.4626, -0.8719,  0.9729, -0.3532,  2.4639],
#         [ 1.4050,  2.3985,  1.1439,  0.1312,  0.8451,  0.6842,  0.2701,  1.0854],
#         [-0.5036, -2.2952,  0.0188,  1.7705,  0.5439,  0.7304, -1.6171,  1.0013]],
print(cute_is_cat)
# tensor([[-0.9291,  1.2762, -0.5389,  1.4626, -0.8719,  0.9729, -0.3532,  2.4639],
#         [ 1.4050,  2.3985,  1.1439,  0.1312,  0.8451,  0.6842,  0.2701,  1.0854],
#         [-0.5036, -2.2952,  0.0188,  1.7705,  0.5439,  0.7304, -1.6171,  1.0013]],


