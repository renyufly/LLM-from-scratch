import tiktoken

# 选择 GPT-2 对应的 分词器(tokenizer)
encoder = tiktoken.get_encoding("gpt2")
print("number of GPT2's token: ", encoder.n_vocab)  # 词表大小-50257

for text in ["cat is cute", "猫可爱"]:
    '''  
                    词表里的位置编号
    "cat is cute"  [9246, 318, 13779]
    "猫可爱"       [163, 234, 104, 20998, 107, 163, 230, 109]
    '''
    # Token ID只是一个索引编号，数值大小不代表语义关系
    token_ids = encoder.encode(text)  # 把文字变成数字
    print(token_ids)

'''
BPE 是tokenizer背后的算法：把经常挨在一起的字节， 合并成一个 Token。
BPE 不知道汉字 和 英文，哪两个字节经常一起出现， 就合并；纯粹是统计，没有任何语言理解
'''