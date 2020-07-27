import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        # Q,V,K = [hid dim, hid dim]
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query = [batch size, src len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        '''
        Q、K、V = [batch size, n heads, query len/src len, head dim]
        这里我们需要将[src len, hid dim] 分解成 n_heads个 [src len, head dim]
        所以不能直接使用 Q.view(batch_size, self.n_heads, src len, head_dim)
        虽然最终形成的矩阵维度一样，但view会破坏原有矩阵的维度信息
        '''
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # attentionMatrix = [batch size, n heads, src len, src len]
        attentionMatrix = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attentionMatrix = attentionMatrix.masked_fill(mask == 0, -1e10)

        # dim=0 对每一行进行softmax, dim=-1 对每一列进行softmax
        # attentionMatrix = F.softmax(attentionMatrix, dim=-1)

        attentionMatrix = torch.softmax(attentionMatrix, dim=-1)

        # x = [batch size, n heads, src len, head dim]
        x = torch.matmul(self.dropout(attentionMatrix), V)

        '''
        x = [batch size, src len, n heads, head dim]
        这里需要使用contiguous让x获得连续的内存，不然下一段代码会报错
        '''
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, src len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attentionMatrix
