import torch
import torch.nn as nn
import math

class inputembedding(nn.Module):

    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class positionencoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        '''
        For even positions : PE(pos,2i) = sin(pos/10000**(2i/dmodel) )
        For odd  positions : PE(pos,2i+1) = cos(pos/10000**(2i/dmodel) )
        where pos is the position in the embedding vector and i is the dimension
        '''

        pe = torch.zeros(seq_len, d_model) # shape = (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.tensor).unsqueeze(1) # shape = (seq_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        '''
        have to add batch dimension to pe
        '''

        pe = pe.unsqueeze(0) # shape = (1, seq_len, d_model)

        self.register_buffer('pe', pe) # pe is not a parameter, so it is registered as buffer, saved in state_dict with same dtype as pe

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1]]).requires_grad_(False) # pe is not a parameter, so it is not updated during backpropagation
        return self.dropout(x)
    
# layer normalisation works indivisually on each dimension of the input tensor
# it normalises the input tensor across the feature dimension
class LayerNormalisation(nn.Module):
    def __init__(self, eps:float=1e-6) -> None:
        super().__init__()
        self.eps = eps

        # there 2 parameters in layer normalisation, gamma/alpha and beta
        # gamma is the scaling parameter (multiplication) and beta is the shifting parameter(addition)

        self.alpha = nn.Parameter(torch.ones(1)) # multiplying factor
        self.beta = nn.Parameter(torch.zeros(1)) # adding factor

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.beta

'''
parameters for feed forward block mentioned in the paper are:
    d_model = 512
    d_ff = 2048
    dropout = 0.1
'''
class feedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1, bias is True by default
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2, bias is True by default
    
    def forward(self,x):
        '''
        fornula for feed forward block is --> FFN(x) = max(0,xW1 + b1)W2 + b2

        (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        '''
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # relu activation function is used
    

'''
parameters for feed forward block mentioned in the paper are:
    d_model = 512
    h = 8
    d_k = d_v = 64 = (d_model/h)
    d_ff = 2048
    dropout = 0.1
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model should be divisible by h"
        self.d_k = d_model // h
        self.d_v = d_model // h

        # the dimention of w is (d_model, d_model), so the input and output dimention of w is d_model
        # Wq, Wk, Wv and Wo are the learnable parameters

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, value, key , mask, dropout:nn.Dropout) -> None:
        '''
        scaled dot product attention
        query, key, value shape = (batch, h, seq_len, d_k)
        mask shape = (batch, seq_len, seq_len)
        '''

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1]) # (batch, h, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = torch.softmax(scores, dim=-1) # (batch, h, seq_len, seq_len)
        
        if dropout is not None:
            scores = dropout(scores)

        return torch.matmul(scores, value), scores


    def forward(self, q, v, k , mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # split the query, key and value into h heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # (batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # (batch, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_v).transpose(1, 2) # (batch, h, seq_len, d_v)

        
        '''Attention(Q,K,V ) = softmax(Q@t.K/√d_k)V '''
        x,self.attention_scores = self.attention(query, value, key, mask, self.dropout) # (batch, h, seq_len, d_v)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], x.shape[1] * x.shape[3]) # (batch, seq_len, d_model)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super().__init__()
        self.norm = LayerNormalisation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:feedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) 

    def forward(self, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers:nn.ModualList):
        super().__init__()

        # list of encoder blocks that contains self attention and feed forward block 
        # with includeing residual connection that contains layer normalisation and dropout
        self.layers = layers 
        self.norm = LayerNormalisation()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention, feed_forward_block:feedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.src_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])


    '''
    parameters:
    x        : target input
    enc_out  : output of the encoder
    src_mask : mask for the source input
    tgt_mask : mask for the target input
    '''

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.src_attention_block(x,enc_out,enc_out,src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModualList):
        super().__init__()

        # list of decoder blocks that contains self attention, cross attention and feed forward block 
        # with includeing residual connection that contains layer normalisation and dropout
        self.layers = layers 
        self.norm = LayerNormalisation()

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)
    

class projecitonlayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.linear(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:inputembedding, tgt_embed:inputembedding, projectionlayer:projecitonlayer, posiiton_encoding:positionencoding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projectionlayer = projectionlayer
        self.position_encoding = posiiton_encoding

    def encoder(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decoder(self, tgt, enc_out, src_mask, tgt_mask):