import numpy as np
import torch
import torch.nn as nn
import os
import random
import pdb

def process_foa_input_sed_doa_labels(feat, label):
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label

def process_foa_input_sed_doa(feat):
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat

def process_foa_input_128d_sed_doa_labels(feat, label):
    mel_bins = 128
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label

def process_foa_input_ssast_data_labels(feat, label):
    mel_bins = 128
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    feat = feat[0, :, :]
    return feat, label


def process_foa_input_sed_labels(feat, label):
    nb_classes = 13
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label[:,:nb_classes]

class SedDoaDistResult():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
    
    def add_item(self, wav_name, sed_pred, doa_pred, dist_pred): #
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        for frame_cnt in range(sed_pred.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt]>0.5:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13], dist_pred[frame_cnt][class_cnt]])

    def add_items(self, wav_names, net_output):
        sed = net_output[:,:,:13]
        doa = net_output[:,:,13:52]
        dist = net_output[:,:,52:]
        if isinstance(sed, torch.Tensor):
            sed = sed.detach().cpu().numpy()
        if isinstance(doa, torch.Tensor):
            doa = doa.detach().cpu().numpy()
        if isinstance(dist,torch.Tensor):
            dist = dist.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, sed[b], doa[b], dist[b])

    def get_result(self):
        return self.output_dict


class SedDoaResult_2023():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
    
    def add_item(self, wav_name, sed_pred, doa_pred): #
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        for frame_cnt in range(sed_pred.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt]>0.5:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])
    
    def add_items(self, wav_names, net_output):
        sed = net_output[:,:,:13]
        doa = net_output[:,:,13:52]
        if isinstance(sed, torch.Tensor):
            sed = sed.detach().cpu().numpy()
        if isinstance(doa, torch.Tensor):
            doa = doa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, sed[b], doa[b])

    def get_result(self):
        return self.output_dict

class SedDoaResult_hop():
    def __init__(self, segment_length, hoplen=10) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
        self.dict = {}
        self.hoplen = hoplen * 10

    def add_item(self, wav_name, seq_result):
        # print(seq_result.shape)
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3]) 
        # start_frame = int(items[-1]) * self.hoplen
        seg_cnt = int(items[-1])
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        self.output_dict[csv_name][seg_cnt] = seq_result
        
    def add_items(self, wav_names, net_output):
        seddoa = net_output
        if isinstance(seddoa, torch.Tensor):
            seddoa = seddoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, seddoa[b])

    def _process(self):
        tmp = {}
        enddic = {}
        for k, v in self.output_dict.items():
            seg_cnt = self.output_dict[k].keys()
            max_cnt = max(seg_cnt)
            # print(max_cnt)
            endframe = max_cnt*self.hoplen + self.segment_length
            # print(endframe)
            tmp[k] = torch.zeros(endframe, 52)
            divide = torch.zeros(endframe, 52)

            for segid, val in v.items():
                # print(val.shape)
                # print('+++++++++++++++++')
                startframe = segid * self.hoplen
                if val.shape[0] != self.segment_length:
                    divide[startframe:startframe+val.shape[0], :] += 1
                    tmp[k][startframe:startframe+val.shape[0], :] += val
                    endframe = startframe+val.shape[0]
                else:
                    divide[startframe:startframe+self.segment_length, :] += 1
                    tmp[k][startframe:startframe+self.segment_length, :] += val
            tmp[k] /= divide
            enddic[k] = endframe
        self.output_dict = tmp
        # print(k, endframe)
        return enddic

    def calres(self):
        enddic = self._process()
        outdir = '/yrfs1/intern/qingwang28/DCASE2022/model_ensemble/Resnet-Conformer-Twotask-new-hop_output'
        for csv_name, val in self.output_dict.items():
            np.save(os.path.join(outdir, '{}.npy'.format(csv_name)), val)
            endframe = enddic[csv_name]
            if csv_name not in self.dict:
                self.dict[csv_name] = {}
            print(csv_name, val.shape, endframe)
            sed_pred = val[:,:13]
            doa_pred = val[:,13:]
            #sed_pred, doa_pred = self.get_sed_doa(val)
            print(sed_pred.shape, doa_pred.shape)
            print('+++++++++++++++++')
            for frame_cnt in range(endframe):
                # items = wav_name.split('_')
                # csv_name = '_'.join(items[:-3])  
                output_dict_frame_cnt = frame_cnt
                # print(sed_pred.shape)
                for class_cnt in range(sed_pred.shape[1]):
                    # print(sed_pred[frame_cnt][class_cnt])
                    if sed_pred[frame_cnt][class_cnt]>0.5:
                        if output_dict_frame_cnt not in self.dict[csv_name]:
                            self.dict[csv_name][output_dict_frame_cnt] = []
                        self.dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])

    def get_result(self):
        return self.dict

class SedResult():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
    
    def add_item(self, wav_name, sed_pred):
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        for frame_cnt in range(sed_pred.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt]>0.5:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, 0, 0, 0])
    
    def add_items(self, wav_names, net_output):
        sed = net_output[:,:,:13]
        doa = net_output[:,:,13:]
        if isinstance(sed, torch.Tensor):
            sed = sed.detach().cpu().numpy()
        if isinstance(doa, torch.Tensor):
            doa = doa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, sed[b], doa[b])

    def get_result(self):
        return self.output_dict

class SedDoaLoss(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:] # torch.Size([32, 100, 52])
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        # sed_label_repeat = sed_label.repeat(1,1,4)
        #pdb.set_trace()
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label) # why multiply with sed_label_repeat? be
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss

class MSPELoss(torch.nn.Module):
    def __init__(self):
        super(MSPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        absolute_error = torch.abs(y_true - y_pred)
        percentage_error = absolute_error / torch.abs(y_true)
        mspe = torch.mean(percentage_error ** 2)
        return mspe
    
class SedLoss_2024(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        sed_label = target[:,:,:13]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        #sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        return loss_sed
    
class SedDistLoss_2024_MSPE(nn.Module):
    def __init__(self, loss_weight=[1.0, 0.1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        # self.criterion_doa = nn.MSELoss()
        self.criterion_dist = nn.MSELoss()
        # self.criterion_dist = MSPELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        dist_out = output[:,:,52:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        dist_label = target[:,:,52:]
        dist_label += 1e-8
        loss_sed = self.criterion_sed(sed_out, sed_label)
        #sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        # sed_label_repeat = sed_label.repeat(1,1,3)
        #pdb.set_trace()
        # loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        # loss_dist = self.criterion_dist(dist_out * sed_label, dist_label)
        loss_dist = self.criterion_dist(dist_out* sed_label/ dist_label, dist_label* sed_label/ dist_label)
        loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
        return loss_sed, loss_dist, loss
    
class SedDistLoss_2024_MAPE(nn.Module):
    def __init__(self, loss_weight=[1.0, 0.1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        # self.criterion_doa = nn.MSELoss()
        # self.criterion_dist = nn.MSELoss()
        self.criterion_dist = nn.L1Loss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        dist_out = output[:,:,52:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        dist_label = target[:,:,52:]
        dist_label += 1e-8
        loss_sed = self.criterion_sed(sed_out, sed_label)
        loss_dist = self.criterion_dist(dist_out* sed_label/ dist_label, dist_label* sed_label/ dist_label)
        loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
        return loss_sed, loss_dist, loss
    
class SedDistLoss_2024_MSE(nn.Module):
    def __init__(self, loss_weight=[1.0, 0.1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        # self.criterion_doa = nn.MSELoss()
        self.criterion_dist = nn.MSELoss()
        # self.criterion_dist = MSPELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        dist_out = output[:,:,52:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        dist_label = target[:,:,52:]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        #sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        # sed_label_repeat = sed_label.repeat(1,1,3)
        #pdb.set_trace()
        # loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        loss_dist = self.criterion_dist(dist_out * sed_label, dist_label)
        loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
        return loss_sed, loss_dist, loss

class SedSCELoss(nn.Module):
    def __init__(self, loss_weight=[1, 2]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        self.criterion_sce = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, output, target):
        sed_out = output[:,:,:13]
        sce_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        sed_label = target[:,:,:13]
        sce_label = target[:,:,13:52]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_sce = self.criterion_sce(sce_out * sed_label_repeat, sce_label)
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_sce
        return loss
