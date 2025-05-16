import numpy as np
import torch
import torch.nn as nn
import os

from .cls_tools.SELD_evaluation_metrics_2024 import distance_between_cartesian_coordinates

def process_foa_input_multi_accdoa_labels(feat, label):
    nb_classes = 13
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    num_track_dummy = 6
    # num_axis = 4
    num_axis = 5
    label = label.reshape(label.shape[0], num_track_dummy, num_axis, nb_classes)
    return feat, label

class MultiAccdoaResult():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.thresh_unify = 15
        self.output_dict = {}

    def get_sed_doa(self, accdoa_in, nb_classes=13):
        threshold = np.array((0.8,  0.8,  0.8, 0.4,  0.3, 0.3,  0.2, 0.4,  0.3,  0.8, 0.5,  0.7, 0.8))
        x0, y0, z0 = accdoa_in[:, :1*nb_classes], accdoa_in[:, 1*nb_classes:2*nb_classes], accdoa_in[:, 2*nb_classes:3*nb_classes]
        sed0 = np.sqrt(x0**2 + y0**2 + z0**2)>0.5
        #sed0 = np.sqrt(x0**2 + y0**2 + z0**2)>threshold
        doa0 = accdoa_in[:, :3*nb_classes]

        x1, y1, z1 = accdoa_in[:, 3*nb_classes:4*nb_classes], accdoa_in[:, 4*nb_classes:5*nb_classes], accdoa_in[:, 5*nb_classes:6*nb_classes]
        sed1 = np.sqrt(x1**2 + y1**2 + z1**2)>0.5
        #sed1 = np.sqrt(x1**2 + y1**2 + z1**2)>threshold
        doa1 = accdoa_in[:, 3*nb_classes: 6*nb_classes]

        x2, y2, z2 = accdoa_in[:, 6*nb_classes:7*nb_classes], accdoa_in[:, 7*nb_classes:8*nb_classes], accdoa_in[:, 8*nb_classes:]
        sed2 = np.sqrt(x2**2 + y2**2 + z2**2)>0.5
        #sed2 = np.sqrt(x2**2 + y2**2 + z2**2)>threshold
        doa2 = accdoa_in[:, 6*nb_classes:]

        return sed0, doa0, sed1, doa1, sed2, doa2

    def determine_similar_location(self, sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
        if (sed_pred0 == 1) and (sed_pred1 == 1):
            if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                    doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
                return 1
            else:
                return 0
        else:
            return 0
    
    def add_item(self, wav_name, seq_result):
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = self.get_sed_doa(seq_result)
        for frame_cnt in range(sed_pred0.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred0.shape[1]):
                # determine whether track0 is similar to track1
                flag_0sim1 = self.determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, self.thresh_unify, 13)
                flag_1sim2 = self.determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, self.thresh_unify, 13)
                flag_2sim0 = self.determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, self.thresh_unify, 13)
                # unify or not unify according to flag
                if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                    if sed_pred0[frame_cnt][class_cnt]>0.5:
                    #if sed_pred0[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+13], doa_pred0[frame_cnt][class_cnt+2*13]])
                    if sed_pred1[frame_cnt][class_cnt]>0.5:
                    #if sed_pred1[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+13], doa_pred1[frame_cnt][class_cnt+2*13]])
                    if sed_pred2[frame_cnt][class_cnt]>0.5:
                    #if sed_pred2[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+13], doa_pred2[frame_cnt][class_cnt+2*13]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    if flag_0sim1:
                        if sed_pred2[frame_cnt][class_cnt]>0.5:
                        #if sed_pred2[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+13], doa_pred2[frame_cnt][class_cnt+2*13]])
                        doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])
                    elif flag_1sim2:
                        if sed_pred0[frame_cnt][class_cnt]>0.5:
                        #if sed_pred0[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+13], doa_pred0[frame_cnt][class_cnt+2*13]])
                        doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])
                    elif flag_2sim0:
                        if sed_pred1[frame_cnt][class_cnt]>0.5:
                        #if sed_pred1[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+13], doa_pred1[frame_cnt][class_cnt+2*13]])
                        doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])

    def add_items(self, wav_names, net_output):
        multi_accdoa = net_output
        if isinstance(multi_accdoa, torch.Tensor):
            multi_accdoa = multi_accdoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, multi_accdoa[b])
            
    def get_result(self):
        return self.output_dict

class MultiAccdoaResult_2024():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.thresh_unify = 15
        self.output_dict = {}

    def get_sed_doa(self, accdoa_in, nb_classes=13):
        threshold = np.array((0.8,  0.8,  0.8, 0.4,  0.3, 0.3,  0.2, 0.4,  0.3,  0.8, 0.5,  0.7, 0.8))
        x0, y0, z0 = accdoa_in[:, :1*nb_classes], accdoa_in[:, 1*nb_classes:2*nb_classes], accdoa_in[:, 2*nb_classes:3*nb_classes]
        sed0 = np.sqrt(x0**2 + y0**2 + z0**2)>0.5
        #sed0 = np.sqrt(x0**2 + y0**2 + z0**2)>threshold
        doa0 = accdoa_in[:, :3*nb_classes]
        dist0 = accdoa_in[:, 3*nb_classes:4*nb_classes]

        x1, y1, z1 = accdoa_in[:, 4*nb_classes:5*nb_classes], accdoa_in[:, 5*nb_classes:6*nb_classes], accdoa_in[:, 6*nb_classes:7*nb_classes]
        sed1 = np.sqrt(x1**2 + y1**2 + z1**2)>0.5
        #sed1 = np.sqrt(x1**2 + y1**2 + z1**2)>threshold
        doa1 = accdoa_in[:, 4*nb_classes: 7*nb_classes]
        dist1 = accdoa_in[:, 7*nb_classes: 8*nb_classes]

        x2, y2, z2 = accdoa_in[:, 8*nb_classes:9*nb_classes], accdoa_in[:, 9*nb_classes:10*nb_classes], accdoa_in[:, 10*nb_classes:11*nb_classes]
        sed2 = np.sqrt(x2**2 + y2**2 + z2**2)>0.5
        #sed2 = np.sqrt(x2**2 + y2**2 + z2**2)>threshold
        doa2 = accdoa_in[:, 8*nb_classes:11*nb_classes]
        dist2 = accdoa_in[:, 11*nb_classes: ]

        return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2

    def determine_similar_location(self, sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
        if (sed_pred0 == 1) and (sed_pred1 == 1):
            if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                    doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
                return 1
            else:
                return 0
        else:
            return 0
    
    def add_item(self, wav_name, seq_result):
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = self.get_sed_doa(seq_result)
        for frame_cnt in range(sed_pred0.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred0.shape[1]):
                # determine whether track0 is similar to track1
                flag_0sim1 = self.determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, self.thresh_unify, 13)
                flag_1sim2 = self.determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, self.thresh_unify, 13)
                flag_2sim0 = self.determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, self.thresh_unify, 13)
                # unify or not unify according to flag
                if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                    if sed_pred0[frame_cnt][class_cnt]>0.5:
                    #if sed_pred0[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+13], doa_pred0[frame_cnt][class_cnt+2*13], dist_pred0[frame_cnt][class_cnt]])
                    if sed_pred1[frame_cnt][class_cnt]>0.5:
                    #if sed_pred1[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+13], doa_pred1[frame_cnt][class_cnt+2*13], dist_pred1[frame_cnt][class_cnt]])
                    if sed_pred2[frame_cnt][class_cnt]>0.5:
                    #if sed_pred2[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+13], doa_pred2[frame_cnt][class_cnt+2*13], dist_pred2[frame_cnt][class_cnt]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    if flag_0sim1:
                        if sed_pred2[frame_cnt][class_cnt]>0.5:
                        #if sed_pred2[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+13], doa_pred2[frame_cnt][class_cnt+2*13], dist_pred2[frame_cnt][class_cnt]])
                        doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13], dist_pred_fc[class_cnt]])
                    elif flag_1sim2:
                        if sed_pred0[frame_cnt][class_cnt]>0.5:
                        #if sed_pred0[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+13], doa_pred0[frame_cnt][class_cnt+2*13], dist_pred0[frame_cnt][class_cnt]])
                        doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13], dist_pred_fc[class_cnt]])
                    elif flag_2sim0:
                        if sed_pred1[frame_cnt][class_cnt]>0.5:
                        #if sed_pred1[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+13], doa_pred1[frame_cnt][class_cnt+2*13], dist_pred1[frame_cnt][class_cnt]])
                        doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13], dist_pred_fc[class_cnt]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                    dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13], dist_pred_fc[class_cnt]])

    def add_items(self, wav_names, net_output):
        multi_accdoa = net_output
        if isinstance(multi_accdoa, torch.Tensor):
            multi_accdoa = multi_accdoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, multi_accdoa[b])
            
    def get_result(self):
        return self.output_dict

class MultiAccdoaResult_thres():
    def __init__(self, segment_length, threshold) -> None:
        self.segment_length = segment_length
        self.thresh_unify = 15
        self.output_dict = {}
        self.threshold = threshold

    def get_sed_doa(self, accdoa_in, nb_classes=13):
        x0, y0, z0 = accdoa_in[:, :1*nb_classes], accdoa_in[:, 1*nb_classes:2*nb_classes], accdoa_in[:, 2*nb_classes:3*nb_classes]
        #sed0 = np.sqrt(x0**2 + y0**2 + z0**2)>0.5
        sed0 = np.sqrt(x0**2 + y0**2 + z0**2)>self.threshold
        doa0 = accdoa_in[:, :3*nb_classes]

        x1, y1, z1 = accdoa_in[:, 3*nb_classes:4*nb_classes], accdoa_in[:, 4*nb_classes:5*nb_classes], accdoa_in[:, 5*nb_classes:6*nb_classes]
        #sed1 = np.sqrt(x1**2 + y1**2 + z1**2)>0.5
        sed1 = np.sqrt(x1**2 + y1**2 + z1**2)>self.threshold
        doa1 = accdoa_in[:, 3*nb_classes: 6*nb_classes]

        x2, y2, z2 = accdoa_in[:, 6*nb_classes:7*nb_classes], accdoa_in[:, 7*nb_classes:8*nb_classes], accdoa_in[:, 8*nb_classes:]
        #sed2 = np.sqrt(x2**2 + y2**2 + z2**2)>0.5
        sed2 = np.sqrt(x2**2 + y2**2 + z2**2)>self.threshold
        doa2 = accdoa_in[:, 6*nb_classes:]

        return sed0, doa0, sed1, doa1, sed2, doa2

    def determine_similar_location(self, sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
        if (sed_pred0 == 1) and (sed_pred1 == 1):
            if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                    doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
                return 1
            else:
                return 0
        else:
            return 0
    
    def add_item(self, wav_name, seq_result):
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = self.get_sed_doa(seq_result)
        for frame_cnt in range(sed_pred0.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred0.shape[1]):
                # determine whether track0 is similar to track1
                flag_0sim1 = self.determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, self.thresh_unify, 13)
                flag_1sim2 = self.determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, self.thresh_unify, 13)
                flag_2sim0 = self.determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, self.thresh_unify, 13)
                # unify or not unify according to flag
                if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                    if sed_pred0[frame_cnt][class_cnt]>0.5:
                    #if sed_pred0[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+13], doa_pred0[frame_cnt][class_cnt+2*13]])
                    if sed_pred1[frame_cnt][class_cnt]>0.5:
                    #if sed_pred1[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+13], doa_pred1[frame_cnt][class_cnt+2*13]])
                    if sed_pred2[frame_cnt][class_cnt]>0.5:
                    #if sed_pred2[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.output_dict[csv_name]:
                            self.output_dict[csv_name][output_dict_frame_cnt] = []
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+13], doa_pred2[frame_cnt][class_cnt+2*13]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    if flag_0sim1:
                        if sed_pred2[frame_cnt][class_cnt]>0.5:
                        #if sed_pred2[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+13], doa_pred2[frame_cnt][class_cnt+2*13]])
                        doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])
                    elif flag_1sim2:
                        if sed_pred0[frame_cnt][class_cnt]>0.5:
                        #if sed_pred0[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+13], doa_pred0[frame_cnt][class_cnt+2*13]])
                        doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])
                    elif flag_2sim0:
                        if sed_pred1[frame_cnt][class_cnt]>0.5:
                        #if sed_pred1[frame_cnt][class_cnt]>threshold[class_cnt]:
                            self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+13], doa_pred1[frame_cnt][class_cnt+2*13]])
                        doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                        self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])
                elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+13], doa_pred_fc[class_cnt+2*13]])

    def add_items(self, wav_names, net_output):
        multi_accdoa = net_output
        if isinstance(multi_accdoa, torch.Tensor):
            multi_accdoa = multi_accdoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, multi_accdoa[b])
            
    def get_result(self):
        return self.output_dict

class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*13]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=13]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss