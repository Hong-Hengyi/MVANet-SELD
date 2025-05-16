import torch
import torch.nn as nn

from .resnet import resnet18, resnet18_nopool, BasicBlock
from .conformer import ConformerBlock

import numpy as np

layer_resnet = ['conv1', 'bn1', 'relu', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'maxpool1', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'maxpool2', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'maxpool3', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'conv5']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 定义音频视觉注意力网络类
class AudioVisualAttentionNetwork_weight(nn.Module):
    def __init__(self, audio_feature_dim, visual_feature_dim, attention_dim):
        super(AudioVisualAttentionNetwork_weight, self).__init__()
        # 音频特征映射到与视觉特征相同的维度
        self.audio_to_common_dim = nn.Sequential(
            nn.Linear(audio_feature_dim, audio_feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(audio_feature_dim, attention_dim),
        )
        # 视觉特征映射到与音频相同的维度
        self.visual_to_common_dim = nn.Sequential(
            nn.Linear(visual_feature_dim, visual_feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(visual_feature_dim, attention_dim),
        )
        self.Visual_weight_calculations = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.Sigmoid()
        )
    
    def forward(self, audio_features, visual_features):
        audio_features = self.audio_to_common_dim(audio_features)
        visual_features = self.visual_to_common_dim(visual_features)
        audio_video_features = audio_features + visual_features
        audio_video_features = torch.tanh(audio_video_features)
        Visual_weight = self.Visual_weight_calculations(audio_video_features)
        return Visual_weight


class ResnetConformer_sed_coord_resnet50_medium_4_guided_m_1_weight_offline(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, benchmark_method=True):
        super().__init__()
        #self.resnet50_model = resnet50()
        #self.resnet50_backbone = torch.nn.Sequential(*(list(self.resnet50_model.children())[:-2]))
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.benchmark_method = benchmark_method
        self.Cross_Modality_Attention_Module_m = AudioVisualAttentionNetwork_weight(audio_feature_dim=512, visual_feature_dim=49, attention_dim=49)
        self.Cross_Modality_Attention_Module_l_1 = AudioVisualAttentionNetwork_weight(audio_feature_dim=256, visual_feature_dim=256, attention_dim=256)
        self.Cross_Modality_Attention_Module_l_2 = AudioVisualAttentionNetwork_weight(audio_feature_dim=256, visual_feature_dim=256, attention_dim=256)
        self.Cross_Modality_Attention_Module_l_3 = AudioVisualAttentionNetwork_weight(audio_feature_dim=256, visual_feature_dim=256, attention_dim=256)
        embedding_dim = 512
        encoder_dim = 256
        self.input_projection_audio = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        embedding_dim_video = 49
        self.input_projection_video = nn.Linear(embedding_dim_video, encoder_dim)

        num_layers_audio_1 = 2
        self.conformer_layers_audio_1 = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
                ) for _ in range(num_layers_audio_1)]
            )

        num_layers_audio_2 = 2
        self.conformer_layers_audio_2 = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
                ) for _ in range(num_layers_audio_2)]
            )

        num_layers_audio_3 = 2
        self.conformer_layers_audio_3 = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
                ) for _ in range(num_layers_audio_3)]
            )

        num_layers_video_1 = 2
        self.conformer_layers_video_1 = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
                ) for _ in range(num_layers_video_1)]
            )

        num_layers_video_2 = 2
        self.conformer_layers_video_2 = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
                ) for _ in range(num_layers_video_2)]
            )

        num_layers_video_3 = 2
        self.conformer_layers_video_3 = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
                ) for _ in range(num_layers_video_3)]
            )

        num_layers = 2
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
                ) for _ in range(num_layers)]
            )

        self.t_pooling = nn.AvgPool1d(kernel_size=5)
        self.t_pooling_2 = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13),
            nn.Sigmoid()
        )
        self.coord_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.Linear(encoder_dim, 39)
        )

    def forward(self, x, vx_res):
        # audio encoder
        a_N, a_C, a_T, a_F = x.shape  # [B, 7, 500, 64]
        audio_outputs = self.resnet(x) # [B, 256, 500, 2)
        N,C,T,W = audio_outputs.shape # (B, 256, 500, 2)
        audio_outputs = audio_outputs.permute(0,2,1,3).reshape(N, T, C*W) # [B, 500, 512]

        N_av, frame_num, C_av, W_av = vx_res.shape # [B, 100, 7, 7]
        vx_res = vx_res.reshape(N_av, frame_num, C_av*W_av)# [B, 100, 49]

        audio_outputs_pool_1 = audio_outputs.permute(0,2,1)
        audio_outputs_pool_1 = self.t_pooling(audio_outputs_pool_1)
        audio_outputs_pool_1 = audio_outputs_pool_1.permute(0,2,1)

        vx_1 = self.Cross_Modality_Attention_Module_m(audio_outputs_pool_1, vx_res) # [B, 100, 49]
        vx_1 = self.input_projection_video(vx_1) # [B, 100, 256]
        audio_outputs_1 = self.input_projection_audio(audio_outputs) # [B, 500, 256]

        for layer_1 in self.conformer_layers_audio_1:
            audio_outputs_1 = layer_1(audio_outputs_1)  # [B, 500, 256]
        for layer_2 in self.conformer_layers_video_1:
            vx_1 = layer_2(vx_1)  # [B, 100, 256]

        audio_outputs_pool_2 = audio_outputs_1.permute(0,2,1)
        audio_outputs_pool_2 = self.t_pooling(audio_outputs_pool_2)
        audio_outputs_pool_2 = audio_outputs_pool_2.permute(0,2,1)

        vx_2 = self.Cross_Modality_Attention_Module_l_1(audio_outputs_pool_2, vx_1)
        audio_outputs_2 = audio_outputs_1

        for layer_3 in self.conformer_layers_audio_2:
            audio_outputs_2 = layer_3(audio_outputs_2)  # [B, 500, 256]
        for layer_4 in self.conformer_layers_video_2:
            vx_2 = layer_4(vx_2)  # [B, 100, 256]

        audio_outputs_pool_3 = audio_outputs_2.permute(0,2,1)
        audio_outputs_pool_3 = self.t_pooling(audio_outputs_pool_3)
        audio_outputs_pool_3 = audio_outputs_pool_3.permute(0,2,1)

        vx_3 = self.Cross_Modality_Attention_Module_l_2(audio_outputs_pool_3, vx_2)
        audio_outputs_3 = audio_outputs_2

        for layer_5 in self.conformer_layers_audio_3:
            audio_outputs_3 = layer_5(audio_outputs_3)  # [B, 500, 256]
        for layer_6 in self.conformer_layers_video_3:
            vx_3 = layer_6(vx_3)  # [B, 100, 256]

        audio_outputs_pool_4 = audio_outputs_3.permute(0,2,1)
        audio_outputs_pool_4 = self.t_pooling(audio_outputs_pool_4)
        audio_outputs_pool_4 = audio_outputs_pool_4.permute(0,2,1)

        vx_4 = self.Cross_Modality_Attention_Module_l_3(audio_outputs_pool_4, vx_3)
        vx_5 = torch.stack([vx_4 for _ in range(a_T // frame_num)], dim=2).reshape(a_N, a_T, C)
        audio_outputs_4 = audio_outputs_3

        conformer_outputs = vx_5 + audio_outputs_4
        for layer_7 in self.conformer_layers:
            conformer_outputs = layer_7(conformer_outputs)  # [B, 500, 256]

        outputs = conformer_outputs.permute(0,2,1)  # [B, 256, 500]
        outputs = self.t_pooling_2(outputs)  # [B, 256, 100]
        outputs = outputs.permute(0,2,1)  # [B, 100, 256]
        sed = self.sed_out_layer(outputs)
        coord = self.coord_out_layer(outputs)
        pred = torch.cat((sed, coord), dim=-1)
        return pred
