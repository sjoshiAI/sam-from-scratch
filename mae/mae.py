from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    def __init__(self, patch_size, encoder_embed_size, decoder_embed_size, image_size, in_channels, masking_ratio, num_encoder_layers, 
                 num_decoder_layers, encoder_num_heads, encoder_head_dim, decoder_num_heads, decoder_head_dim):
        super().__init__()
        self.patch_size = patch_size
        self.encoder_embed_size = encoder_embed_size
        self.decoder_embed_size = decoder_embed_size
        self.in_channels = in_channels
        self.masking_ratio = masking_ratio
        self.num_patches = (image_size//patch_size)**2
        self.patch_embedding = PatchEmbedding(patch_size, self.encoder_embed_size, image_size, in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_size)) # shared class token so that's why added as 1,1 
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.encoder_pe = PositionalEmbeddingSinCos(self.num_patches + 1, self.encoder_embed_size)
        self.decoder_pe = PositionalEmbeddingSinCos(self.num_patches + 1, self.decoder_embed_size)
        self.enc_to_dec = nn.Linear(self.encoder_embed_size, self.decoder_embed_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_size))
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_num_heads = encoder_num_heads
        self.encoder_head_dim = encoder_head_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_head_dim = decoder_head_dim
        self.encoder = nn.ModuleList([TransformerEncoder(self.encoder_embed_size, self.encoder_head_dim, self.encoder_num_heads) for _ in range(self.num_encoder_layers)])
        self.decoder = nn.ModuleList([TransformerEncoder(self.decoder_embed_size, self.decoder_head_dim, self.decoder_num_heads) for _ in range(self.num_decoder_layers)])
        self.reconstruction_head = nn.Linear(self.decoder_embed_size, self.patch_size*self.patch_size*self.in_channels)
    
    def forward(self, x):
        B,_,_,_ = x.shape
        x = self.patch_embedding(x)
        
        ## Logic to handle the class tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.encoder_pe(x)
        cls_token = x[:,0:1,:]
        x = x[:,1:,:]
        
        ## Logic to remove masking ratio % of tokens to feed to encoder
        # Generate random indices
        noise = torch.rand(B,self.num_patches, device=x.device)
        indices = torch.argsort(noise, dim=1)
        num_keep = int(self.num_patches*(1-self.masking_ratio))
        keep_indices = indices[:,:num_keep]
        keep_indices_expanded = keep_indices.unsqueeze(-1).expand(-1,-1,self.encoder_embed_size)
        visible_patches = torch.gather(x, dim=1, index=keep_indices_expanded)
        x = torch.cat([cls_token, visible_patches], dim=1)
        
        ## Encode the visible patches only
        for block in self.encoder:
            x = block(x)

        ## Project to decoder embedding dim
        encoded = self.enc_to_dec(x)
        ## remove the cls token to be added later
        encoded_cls_token = encoded[:,0:1, :]
        encoded_patches = encoded[:,1:, :]

        ## Add masked patches to the above encoded patches for decoder
        full_patches = self.mask_token.expand(B, self.num_patches, self.decoder_embed_size).clone()
        keep_indices_expanded_dec = keep_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_size)
        full_patches.scatter_(dim=1, index = keep_indices_expanded_dec, src=encoded_patches)
        
        ## add the cls token again
        x = torch.cat([encoded_cls_token, full_patches], dim=1)
        
        ## add decoder positional embedding
        x = self.decoder_pe(x)

        ## apply the decoder
        for block in self.decoder:
            x = block(x)
        decoder_output = x[:,1:,:] # remove cls token now
        x = self.reconstruction_head(decoder_output)
        mask_indices = indices[:, num_keep:]
        return x, mask_indices


class PatchEmbedding(nn.Module):
    """break an image into non-overlapping patches, apply an embedding and 
    returns the output in th form of 
    Batch x num_patches x flattened patch length """
    def __init__(self, patch_size, embed_size, image_size, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_patches = (image_size//patch_size)**2 # assuming square images
        self.patch_embedding = nn.Linear(patch_size*patch_size*in_channels, self.embed_size)

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(x)
        return x

class PositionalEmbeddingSinCos(nn.Module):
    """
    PositionalEmbedding sin cos version from attention is all you need
    """
    def __init__(self,num_patches ,embed_size):
        super().__init__()
        self.embed_size = embed_size
        # Create PE tensor as local variable first
        pe = torch.zeros(num_patches, embed_size)
        pos_indices = torch.arange(num_patches).unsqueeze(1) # column vector
        dim_indices = (torch.arange(self.embed_size)//2).unsqueeze(0)
        denominator = 10000**(2*dim_indices/self.embed_size)
        angles = pos_indices/denominator
        pe[:,0::2] = torch.sin(angles[:, 0::2])
        pe[:,1::2] = torch.cos(angles[:, 1::2])

        # Register as buffer so it moves to device with model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has a shape of B x num_patches x embed_size
        return x + self.pe
        


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
    

def patchify(images, patch_size):                                                                                                                                                              
      """                                                                                                                                                                                        
      Convert images to patches for computing loss                                                                                                                                               
                                                                                                                                                                                                 
      Args:                                                                                                                                                                                      
          images: (B, C, H, W) - input images                                                                                                                                                    
          patch_size: int - size of each patch                                                                                                                                                   
                                                                                                                                                                                                 
      Returns:                                                                                                                                                                                   
          patches: (B, num_patches, patch_size*patch_size*C)                                                                                                                                     
      """                                                                                                                                                                                        
      patches = rearrange(                                                                                                                                                                       
          images,                                                                                                                                                                                
          'b c (h p1) (w p2) -> b (h w) (c p1 p2)',                                                                                                                                              
          p1=patch_size,                                                                                                                                                                         
          p2=patch_size                                                                                                                                                                          
      )                                                                                                                                                                                          
      return patches

def mae_loss(pred, target, mask_indices):
    """                                                                                                                                                                                        
      Compute MAE reconstruction loss only on masked patches                                                                                                                                     
                                                                                                                                                                                                 
      Args:                                                                                                                                                                                      
          pred: (B, num_patches, patch_pixels) - model predictions for all patches                                                                                                               
          target: (B, num_patches, patch_pixels) - ground truth patches                                                                                                                          
          mask_indices: (B, num_masked) - indices of masked patches                                                                                                                              
                                                                                                                                                                                                 
      Returns:                                                                                                                                                                                   
          loss: scalar tensor - mean squared error on masked patches only                                                                                                                        
      """    
    B, num_patches, patch_pixels = pred.shape
    mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, patch_pixels)

    pred_masked = torch.gather(pred, dim=1, index=mask_indices_expanded)
    target_masked = torch.gather(target, dim=1, index=mask_indices_expanded)

    loss = F.mse_loss(pred_masked, target_masked)

    return loss