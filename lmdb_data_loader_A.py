import os
import pdb
import numpy as np
import lmdb
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from utils.lmdb_tools.datum_pb2 import SimpleDatum
from tqdm import tqdm

class LmdbDataset(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        #self.visial_tools = VisualTools()
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            #pdb.set_trace()
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)

            wav_name = datum.wave_name.decode()
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)
        return {'data': data, 'label':label, 'wav_name':wav_name}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]
        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)
        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names
        return out




class LmdbDataset_av(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.av_feat_dir = '../DCASE2024_SELD_dataset/seld_feat_label/video_dev'
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []   
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:  
            lines = f.readlines()
            for k in lines:  # 遍历keys.txt中的每一行
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split:     
                    self.keys.append(k.strip())  
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False) 
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            wav_name = datum.wave_name.decode()
            av_name = wav_name.split("_seg_1")[0]
            av_feat = np.load(os.path.join(self.av_feat_dir, av_name+'.npy'))
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)
        return {'data': data, 'label':label, 'wav_name':wav_name, 'av_feat':av_feat}

    def collater(self, samples):   
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]
        av_feats = [s['av_feat'] for s in samples]
        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)
        collated_av_feats = np.stack(av_feats, axis=0)
        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names
        out['av_feat'] = torch.from_numpy(collated_av_feats)
        return out


class LmdbDataset_logmel_resnet50(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None,
                 data_process_fn=None) -> None:
        super().__init__()
        self.lmdb_dir = lmdb_dir
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []
        #print('1')
        self.use_benchmark_method = True

        # 加载视频特征
        self.video_npy = '/disk3/hyhong/3D_SELD_2024/DCASE2024_SELD_dataset/seld_feat_label/video_dev'
        self.v_feat_all={}
        ##########初始化时只加载所有npy文件名称######
        for filename in tqdm(os.listdir(self.video_npy)):
            video_fea_path = os.path.join(self.video_npy, filename)
            self.v_feat_all[os.path.splitext(filename)[0]] = video_fea_path # 把所有的视频文件保存下来
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split:  # check which split the file belongs to
                    self.keys.append(k.strip())

        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        self.data_cat_dict = {"2": "dev-test-tau",
                            "4": "dev-train-tau",
                            "6": "dev-train-tau",
                            "7": "dev-train-tau",
                            "8": "dev-test-tau",
                            "9": "dev-train-tau",
                            "10": "dev-test-tau",
                            "12": "dev-train-tau",
                            "13": "dev-train-tau",
                            "14": "dev-train-tau",
                            "15": "dev-test-tau",
                            "16": "dev-test-tau",
                            "21": "dev-train-sony",
                            "22": "dev-train-sony",
                            "23": "dev-test-sony",
                            "24": "dev-test-sony"}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        visual_bbox_np = None
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum = SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            wav_name = datum.wave_name.decode()
            label_shape = label.shape[0]
            pad_width = 0
            if self.segment_len is not None and label.shape[0] < self.segment_len:  # training
                pad_width = self.segment_len - label.shape[0]
                data = np.pad(data, pad_width=((0, self.segment_len * 5 - data.shape[0]), (0, 0)))  # 把data的shape变成(500,448)
                label = np.pad(label, pad_width=((0, self.segment_len - label.shape[0]), (0, 0)))  # 把label的shape变成(100,65)

            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)  # reshape: 把data的shape变成(7,500,64)，label不变

        data_cat_num = wav_name.split("_")[1].split("room")[1]
        data_cat_num = wav_name.split("_")[1].split("room")[1]
        _filename = wav_name.split("_seg")[0]  # 'fold3_room12_mix004'
        split_info = wav_name.split("seg_")[-1]  # '34 33'  34代表这段视频一共时长34*10s，33代表当前片段为第33*10s~34*10s
        total_seg = int(split_info.split("_")[0])
        select_seg = int(split_info.split("_")[1])  # 33
        ###########初始化时只加载所有npy文件名称，在getitem中读取特征######
        video_feature = np.load(self.v_feat_all[_filename]).astype(np.float32)
        video_frame_feature = video_feature[select_seg*10*10:(select_seg+1)*10*10, :, :]
        frame_num = video_frame_feature.shape[0]
        if frame_num != 100:
            pad_num = 10*10 - frame_num
            pad_data = np.zeros((pad_num,7,7), dtype=np.float32)  # 不足100的补0
            video_frame_feature = np.concatenate((video_frame_feature,pad_data),axis=0)
        if video_frame_feature.shape[0] != 100:
            print('get_item 视频特征失败！')

        return {'data': data, 'label': label, 'wav_name': wav_name, "visual_bbox": visual_bbox_np,
                "visual_keypoint": video_frame_feature, 'pad_width': pad_width}

    def collater(self, samples):   
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]
        av_feats = [s['visual_keypoint'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)
        collated_av_feats = np.stack(av_feats, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names
        out['av_feat'] = torch.from_numpy(collated_av_feats)

        return out
    
    
    '''
    def collater(self, samples):  # 用在dataloader里面
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        out = {}
        visual_keypoint = [s["visual_keypoint"] for s in samples]
        out["visual_keypoints"] = torch.from_numpy(np.array(visual_keypoint))

        wav_names = [s['wav_name'] for s in samples]
        pad_width = [s['pad_width'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)
        collated_pad_width = np.stack(pad_width, axis=0)
        out['wav_names'] = wav_names
        out['input'] = torch.from_numpy(collated_feats)  # 音频
        out['target'] = torch.from_numpy(collated_labels)
        out['pad_width'] = torch.from_numpy(collated_pad_width)
        return out
    '''




