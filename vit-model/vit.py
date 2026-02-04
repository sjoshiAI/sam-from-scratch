import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image

class LayerNorm(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.gamma = nn.Parameter(torch.ones(self.embed_size))
        self.beta = nn.Parameter(torch.zeros(self.embed_size))

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=-1, keepdim=True) #temporary variables
        eps = 1e-5
        x = (x- mean)/(var**0.5 + eps)  # avoid zero division and provide numerical stability
        x = self.gamma * x + self.beta
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, head_dim = 10, embed_size = 50):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_size = embed_size
        self.total_dim = self.num_heads*self.head_dim
        self.W_q = nn.Linear(self.embed_size, self.total_dim)
        self.W_k = nn.Linear(self.embed_size, self.total_dim)
        self.W_v = nn.Linear(self.embed_size, self.total_dim)
        self.W_o = nn.Linear(self.total_dim, self.embed_size)

    def forward(self, x):
        B, N, _ = x.shape # batch_size, num_patches, embed_size
        q,k,v = self.W_q(x), self.W_k(x), self.W_v(x) ## B x N x total_dim

        q = q.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3) # B x NH x N x HD

        A = F.softmax((q@k.transpose(-2,-1))/ self.head_dim**0.5, dim=-1) # B x NH x N x N. 
        out = A@v ## B x NH x N X total_dim
        out = out.permute(0, 2, 1, 3).contiguous() # B x N x NH x D
        out = out.view(B, N, self.total_dim) # B x N x NH*D

        return self.W_o(out) # B x N x E

class MLP(nn.Module):
    def __init__(self, input_embed_size=50):
        super().__init__()
        self.hidden_dim = 4*input_embed_size
        self.embed_size = input_embed_size
        self.layer1 = nn.Linear(self.embed_size, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.embed_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, head_dim, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(self.num_heads, self.head_dim, self.embed_size)
        self.ln1 = LayerNorm(self.embed_size)
        self.ln2 = LayerNorm(self.embed_size)
        self.mlp = MLP(self.embed_size)

    def forward(self, x):
        x = x + self.mha(self.ln1(x)) # residual connecton
        x = x + self.mlp(self.ln2(x)) # residual connection
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_size):
        super().__init__()
        self.num_tokens = seq_len + 1
        self.embed_size = embed_size
        self.pe = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_size))
        nn.init.trunc_normal_(self.pe, std=0.02) # ijnitialise to match the paper, try with or without this
    
    def forward(self, x):
        x = self.pe + x
        return x

class VisionTransformer(nn.Module):
    def __init__(self, patch_size = 4, embed_size=192, image_size = 32, in_channels=3, num_classes=10, num_heads = 8, head_dim=24, num_layers=6):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_patches = (image_size//patch_size)**2
        self.patch_embed = nn.Linear(patch_size*patch_size*in_channels, embed_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_size)) #extra 1 to match the batch size
        nn.init.trunc_normal_(self.cls_token, std=0.02) # initialise randomly the class token
        self.pe = PositionalEmbedding(self.num_patches, self.embed_size)
        self.transformers = nn.ModuleList([TransformerEncoder(self.embed_size, self.head_dim, self.num_heads) for _ in range(num_layers)])
        self.ln = LayerNorm(self.embed_size)
        self.classifier_head = nn.Linear(self.embed_size, self.num_classes)


    def forward(self, x):
        B,C,H,W = x.shape
        pad_h = (self.patch_size - H%self.patch_size)%self.patch_size
        pad_w = (self.patch_size - W%self.patch_size)%self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pe(x)
        for layer in self.transformers:
            x = layer(x)
        
        x = self.ln(x)
        x = x[:,0] # extract the cls_token added initially
        x = self.classifier_head(x)
        return x






        


