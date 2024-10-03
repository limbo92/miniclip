from torch import nn
import torch
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self,num_embedding:int, embedding_dim:int, output_dim:int):
        super().__init__()
        self.emb = nn.Embedding(num_embedding,embedding_dim)
        self.dense1 = nn.Linear(embedding_dim,out_features=64)
        self.dense2 = nn.Linear(64, out_features=16)
        self.wt = nn.Linear(16, out_features=8)
        self.ln = nn.LayerNorm(normalized_shape=output_dim)

    def forward(self,x):
         x = self.emb(x)
         x = F.relu(self.dense1(x))
         x = F.relu(self.dense2(x))
         x = self.wt(x)
         #print(x)
         x = self.ln(x)
         return x


if __name__ == '__main__':
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    te = TextEncoder(num_embedding=10, embedding_dim=16, output_dim=8)
    te_result = te(x)
    print(te_result)


