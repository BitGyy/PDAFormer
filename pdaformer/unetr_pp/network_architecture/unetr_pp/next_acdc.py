from torch import nn
from typing import Tuple, Union
import torch.nn.functional as F
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.unetr_pp.dynunet_block import UnetOutBlock, UnetResBlock, skip_fussion, cat_out, upsample, Pool
from unetr_pp.network_architecture.unetr_pp.model_components import UnetrUpBlock,NextEncoder,NextUpBlock,MHCUpBlock,Pool,ConvAttn


class UNETR_Next(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [128, 128, 128],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            proj_size = [256,128,64,32],
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "batch",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        Examples::
            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
#         depths = [2,2,2,2]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (1, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = NextEncoder(dims=dims, depths=depths, num_heads=num_heads, proj_size=proj_size)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = NextUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            num_heads=num_heads[3],
            proj_size=64,
            out_size=4*10*10,
        )
        self.decoder4 = NextUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            num_heads=num_heads[2],
            proj_size=128,
            out_size=8 * 20 * 20,
        )
        self.decoder3 = NextUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            num_heads=num_heads[1],
            proj_size=256,
            out_size=16 * 40 * 40,
        )
        self.decoder2 = NextUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(1, 4, 4),
            norm_name=norm_name,
            num_heads=num_heads[0],
            proj_size=64,
            out_size=16 * 160 * 160,
            conv_decoder=True,
        )
        
#         self.skipfuss = skip_fussion(ch = feature_size*2)
        self.skipfuss1 = skip_fussion(i = 2, ch = feature_size*2)
        self.skipfuss2 = skip_fussion(i = 1, ch = feature_size*2)
        self.skipfuss3 = skip_fussion(i = 0, ch = feature_size*2)
        
        # self.catout1 = cat_out(feature_size, feature_size*2, last_layer=True)
        # self.catout2 = cat_out(feature_size*2, feature_size*4, last_layer=False)
        # self.catout3 = cat_out(feature_size*4, feature_size*8, last_layer=False)
        
        # self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size + feature_size*2, out_channels=out_channels)
        # if self.do_ds:
        #     self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2 + feature_size * 4, out_channels=out_channels)
        #     self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4 + feature_size * 8, out_channels=out_channels)
#         self.catout3 = cat_out(in_ch=feature_size*4, in_ch2 = feature_size*8)
#         self.catout2 = cat_out(in_ch=feature_size*2, in_ch2 = feature_size*4)
#         self.catout1 = cat_out(in_ch=feature_size, in_ch2 = feature_size*2, last_layer=True)
        
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
            
        
            
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    
    def forward(self, x_in):
        # print(x_in.shape)
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size) # 256 4 4 4 
        
        fuss1 = [enc1,enc2,enc3,dec4]
        enc3 = self.skipfuss1(fuss1)
#         print('enc3',enc3.shape)
#         skip_feature = self.skipfuss1(fuss1)
        
#         enc3 = skip_feature[2]
#         enc2 = skip_feature[1]
#         enc1 = skip_feature[0]
        
        dec3 = self.decoder5(dec4, enc3) # 128 8 8 8
        
        fuss2 = [enc1,enc2,dec3,dec4]
        enc2 = self.skipfuss2(fuss2)
#         print('dec2',dec2.shape)

        dec2 = self.decoder4(dec3, enc2) # 64 16 16 16
        
        fuss3 = [enc1,dec2,dec3,dec4]
        enc1 = self.skipfuss3(fuss3)
#         print('dec1',dec1.shape)

        dec1 = self.decoder3(dec2, enc1) # 32 32 32 32 
        out = self.decoder2(dec1, convBlock) # 2 16 64 128 128
        
        # dec2_out = self.catout3(dec2,dec3)
        # dec1_out = self.catout2(dec1,dec2)
        # out = self.catout1(out,dec1)
#         out2 = self.catout2(dec1,dec2)
#         out3 = self.catout3(dec2,dec3)
        
        
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            # logits = [self.out1(out), self.out2(dec1_out), self.out3(dec2_out)]
            # logits = [self.out1(out), self.out2(out2), self.out3(out3)]
        else:
            logits = self.out1(out)

        return logits
    

    
    