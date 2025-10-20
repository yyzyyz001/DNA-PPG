import torch
import torch.nn as nn
from pos_embed import tAPE

from timm.models.layers import to_2tuple

class PatchEmbed_ts(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, ts_len=1250, 
                 patch_size=10, 
                 embed_dim=768, 
                 stride=10,
                 dropout=0.1, # position_embed params
                 scale_factor=1.0,):
        super().__init__()

        '''
        For pretrain:
        输入 batch*in_channel*L
        '''

        self.ts_len = ts_len
        self.patch_size = patch_size
        
        self.proj = nn.Conv1d(in_channels=1,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)

        bs, E, P = self.get_output_shape(ts_len) # n, emb_dim, P

        self.patch_hw = patch_size
        self.num_patches = P
        
    def get_output_shape(self, ts_len):
        return self.proj(torch.randn(1,1,ts_len)).shape # bs, num_parches, L

    def forward(self, x):
        bs, L = x.shape
        x = x.unsqueeze(1)  # bs, 1, L

        x = self.proj(x) # bs, E, L
        x = x.permute(0, 2, 1) # bs, L, E

        return x



if __name__ == '__main__':
    # patch_emb = PatchEmbed_new(img_size=(387,65), patch_size=(9,5), in_chans=3, embed_dim=64, stride=(9,5))
    # input = torch.rand(8,3,387,65)
    # output = patch_emb(input)
    # print(output.shape) # (8,559,64)

    # patch_emb = PatchEmbed3D_new(video_size=(6,224,224), patch_size=(2,16,16), in_chans=3, embed_dim=768, stride=(2,16,16))
    # input = torch.rand(8,3,6,224,224)
    # output = patch_emb(input)
    #print(output.shape) # (8,64)

    patch_emb = PatchEmbed_ts(ts_len=1250,patch_size=10,stride=10)
    input = torch.randn(16,1250)
    output = patch_emb(input)
    print(output.shape)
    print(patch_emb.patch_size)