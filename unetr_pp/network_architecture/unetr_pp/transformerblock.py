import torch.nn as nn
import torch
from unetr_pp.network_architecture.unetr_pp.dynunet_block import UnetResBlock,get_conv_layer,MHCResBlock


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
    
class NextBlock_abl(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")


        
        self.epa_block = EPAN(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        

    def forward(self, x):
        
        attn_skip = self.epa_block(x)
        
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
    
class NextBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            # insize: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")
           
        # self.insize = insize
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPAN(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,num_heads=num_heads,channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        
        self.MCHA = MHCResBlock(3, hidden_size, hidden_size, kernel_size=1, stride=1,
                             norm_name="batch",groups = hidden_size//32)
        self.pool = nn.Linear(input_size,1)

        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(hidden_size, hidden_size//num_heads, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(hidden_size//num_heads, hidden_size, bias=False),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(dropout_rate)
        
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
    def forward(self, x):
        B, C, H, W, D = x.shape
        
        
        # x_ca = self.MCHA(x)
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        

        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.norm(x)   
        x_SA =  self.epa_block(x)   
        
        x_ca = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        x_ca = self.MCHA(x_ca)
        # x_ca = x_ca.reshape(B, C, H * W * D).permute(0, 2, 1)
        
        
        x_ca = x_ca.reshape(B, C, H * W * D)
        
        x_CA = self.pool(x_ca).transpose(-1,-2)

        x_CA = self.fc(x_CA)
        attn_CA = self.attn_drop(x_CA)

        x_CA = x_ca.transpose(-1,-2) * attn_CA
        # x_CA = x_CA.reshape(B,C,H * W * D).transpose(-1,-2)
        # print(x.shape)
        # print(x_SA.shape)
        # print(x_CA.shape)
        
        attn = self.gamma * x_SA + self.gamma2 * x_CA + x
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        
      
        
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)
        return x
    
class NextBlock_sa(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            # insize: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")
           
        # self.insize = insize
        
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPAN(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,num_heads=num_heads,channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
#         self.MC_proj = get_conv_layer(3, hidden_size, hidden_size//2, kernel_size=(1,1,1), stride=(0,0,0),)
       
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="instance")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
#         if pos_embed:
#             self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        attn =  x + self.gamma * self.epa_block(self.norm(x))   
       

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        
         
#         attn_skip = torch.cat((attn_skip,x_ca),dim = 1)
        
        
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)
    
#         attn = self.conv51(x)
#         x = x + self.conv8(attn)

        return x
    
class NextBlock_1(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            # insize: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")
           
        # self.insize = insize
        
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPAN(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,num_heads=num_heads,channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        # self.MC_proj = get_conv_layer(3, hidden_size, hidden_size//2, kernel_size=(1,1,1), stride=(0,0,0),)
        self.MCHA = MHCResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1,
                             norm_name="instance",groups=hidden_size//32, )
        
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="instance")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        attn =  x + self.gamma * self.epa_block(self.norm(x))   
        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        x_ca = self.MCHA(x)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        
        attn_skip = attn_skip + x_ca
        # attn_skip = torch.cat((attn_skip,x_ca),dim = 1)
        # attn_skip = x +  attn_skip
        
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)
        return x
    
class NextBlock_cat(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            r: float=0.75,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.sa_channel = int(hidden_size*r)
        self.ca_channel = int(hidden_size*(1-r))
            
        self.pros =  get_conv_layer(3, hidden_size, self.sa_channel, kernel_size=(1,1,1),)
        self.norm = nn.LayerNorm(self.sa_channel)
        self.gamma = nn.Parameter(1e-6 * torch.ones(self.sa_channel), requires_grad=True)
        self.epa_block = EPAN(input_size=input_size, hidden_size=self.sa_channel, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        # self.MC_proj = get_conv_layer(3, hidden_size, hidden_size//2, kernel_size=(1,1,1), stride=(0,0,0),)
        self.proc =  get_conv_layer(3,self.sa_channel, self.ca_channel, kernel_size=(1,1,1),)
        self.MCHA = MHCResBlock(3, self.ca_channel, self.ca_channel, kernel_size=3, stride=1,
                             norm_name="instance",groups=hidden_size//32, )
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="instance")
        # self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, self.sa_channel))

    def forward(self, x):
        x = self.pros(x)
        B, C, H, W, D = x.shape
        
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        attn = x + self.gamma * self.epa_block(self.norm(x))   
        x = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        
        # attn = x + self.gamma * self.epa_block(self.norm(x))

        # attn_skip = attn.reshape(B, H, W, D, C//2).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        
        # attn_skip = attn_skip + x_ca
        x_ca = self.proc(x)
        x_ca = self.MCHA(x_ca)
        x = torch.cat((x,x_ca),dim = 1)
        # print(x.shape)
        # print(attn_skip.shape)
        # print((self.gamma * attn_skip).shape)
        # attn_skip = x +   attn_skip
        
        x = self.conv51(x)
        # x = attn_skip + self.conv8(attn)
        x = self.conv8(x)
        return x
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
class NextBlock_gate(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPAN(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        # self.MC_proj = get_conv_layer(3, hidden_size, hidden_size//2, kernel_size=(1,1,1), stride=(0,0,0),)
        
        self.MCHA = MHCResBlock(3, hidden_size, hidden_size, kernel_size=1, stride=1,
                             norm_name="instance",groups=hidden_size//32, )
        
        self.local_embedding = get_conv_layer(3,hidden_size, hidden_size, kernel_size=(1,1,1), )
        self.global_embedding =  get_conv_layer(3,hidden_size, hidden_size, kernel_size=(1,1,1), )
        self.global_act = get_conv_layer(3,hidden_size, hidden_size, kernel_size=(1,1,1), )
        self.act = h_sigmoid()
        
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="instance")
        # self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        x_sa = x + self.gamma * self.epa_block(self.norm(x))   
        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        
        # attn = x + self.gamma * self.epa_block(self.norm(x))

        x_sa = x_sa.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        x_ca = self.MCHA(x)
        
        x_local = self.local_embedding(x_ca)
        x_global = self.global_embedding(x_sa)
        x_act = self.act(self.global_act(x_sa))
        
        
        
        x = x_local*x_act + x_global
        
        x = self.conv51(x)
        # x = attn_skip + self.conv8(attn)
        x = self.conv8(x)
        return x
    
class EPAN_ori(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))
        
        

    def forward(self, x):
        B, N, C = x.shape
        

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        # print(qkvv.shape)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
    
class EPAN_sel(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))
        
        

    def forward(self, x):
        B, N, C = x.shape
        

        qkvv = self.qkvv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # print(qkvv.shape)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared,  v_SA = qkvv[0], qkvv[1], qkvv[2]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = x.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (v_CA @ x) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
#         print(attn_CA.shape)

        x_CA = (attn_CA @ x.transpose(-2, -1)).permute(0, 2, 1)
#         print(x_CA.shape)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
#         print(x_SA.shape)
        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
    
class EPAN_3d(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)
        self.pool = nn.Linear(input_size,1)
        self.MCHA = MHCResBlock(3, hidden_size, hidden_size, kernel_size=1, stride=1,
                             norm_name="instance",groups=hidden_size//32, )
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(hidden_size, hidden_size // num_heads, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(hidden_size // num_heads, hidden_size, bias=False),
            nn.Sigmoid()
        )


        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))
        
        

    def forward(self, x):
        B, C, H, W, D = x.shape
        N = H*W*D
        x_c = self.MCHA(x)

        x = x.reshape(B, C, N).permute(0, 2, 1)
        
        x_c = x_c.reshape(B, C, N).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
            x_c = x_c + self.pos_embed
        
#         B, N, C = x.shape
        x = self.norm(x)
        x_c = self.norm(x_c)

        qkvv = self.qkvv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # print(qkvv.shape)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared,  v_SA = qkvv[0], qkvv[1], qkvv[2]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        x_CA = x_c.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)
        
        x_CA = self.pool(x_CA)
        x_CA = self.fc(x_CA.transpose(-2, -1))


        attn_CA = self.attn_drop(x_CA)
#         print(attn_CA.shape)

        x_CA = x_c * attn_CA
#         print(x_CA.shape)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
#         print(x_SA.shape)
        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        
        x_att = torch.cat((x_SA, x_CA), dim=-1)
        
        x = x + self.gamma * x_att

        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}

class EPAN_ca(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        # self.insize = insize
#         self.pos_embed = None
        
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        self.pool = nn.Linear(input_size,1)
        self.MCHA = MHCResBlock(3, hidden_size, hidden_size, kernel_size=1, stride=1,
                             norm_name="instance",groups=hidden_size//32, )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(hidden_size, hidden_size // num_heads, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(hidden_size // num_heads, hidden_size, bias=False),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(spatial_attn_drop)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        
        
        

    def forward(self, x):
        B, C, H, W, D = x.shape
        N = H*W*D
        
        x_c = self.MCHA(x)
#         print(x.shape)

        x = x.reshape(B, C, N).permute(0, 2, 1)
        
        x_c = x_c.reshape(B, C, N).permute(0, 2, 1)
        
        if self.pos_embed is not None:
#             print(x.shape)
#             print(self.pos_embed.shape)
            x = x + self.pos_embed
            x_c = x_c + self.pos_embed
            
        x = self.norm(x)
        x_c = self.norm(x_c)
        x_CA = x_c.transpose(-2, -1)

        x_CA = self.pool(x_CA)
        x_CA = self.fc(x_CA.transpose(-2, -1))
        attn_CA = self.attn_drop(x_CA)
#         print(attn_CA.shape)

        x_CA = x_c * attn_CA
    
        x = x + self.gamma * x_CA

        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
    
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return { 'temperature2'}
    
class EPAN(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        # self.insize = insize
        self.num_heads = num_heads
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        
        

        # self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        
        
        

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        # x = x.reshape(B, self.insize, self.insize, self.insize, C).permute(0, 4, 1, 2, 3)
        # x_ca = self.MCHA(x)
        # print(qkv.shape)

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_SA = qkv[0], qkv[1], qkv[2]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
     
        v_SA = v_SA.transpose(-2, -1)
        # print(k_shared.shape)
        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
       

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        # x_SA = self.out_proj(x_SA)
       
        # x = torch.cat((x_SA, x_CA), dim=-1)
        return x_SA

    @torch.jit.ignore
    def no_weight_decay(self):
        return { 'temperature2'}