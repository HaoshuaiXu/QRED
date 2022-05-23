import torch
from torch import nn
from gensim.models import Word2Vec


class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        # embedding
        wvmodel = Word2Vec.load("/home/XuHaoshuai/Project/HumanIE-IPM-experiment-2.0/word2vec/word2vec.model")
        vocab_size = len(wvmodel.wv)
        vector_size = wvmodel.vector_size
        weight = torch.randn(vocab_size, vector_size)
        words = wvmodel.wv.index_to_key
        word_to_idx = {word: i for i, word in enumerate(words)}
        idx_to_word = {i: word for i, word in enumerate(words)}
        for i in range(len(wvmodel.wv.index_to_key)):
            try:
                index = word_to_idx[wvmodel.wv.index_to_key[i]]
            except:
                continue
        vector=wvmodel.wv.get_vector(idx_to_word[word_to_idx[wvmodel.wv.index_to_key[i]]])
        weight[index, :] = torch.from_numpy(vector)
        self.embedding = nn.Embedding.from_pretrained(weight)
        
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs