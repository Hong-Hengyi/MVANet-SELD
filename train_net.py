import os, shutil, argparse
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import yaml
from lmdb_data_loader_A import LmdbDataset_av, LmdbDataset_logmel_resnet50
from models.resnet_conformer_audio import ResnetConformer_sed_coord_resnet50_medium_4_guided_m_1_weight_offline
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from utils.write_csv import write_output_format_file
from utils.cls_tools.cls_compute_seld_results_2024 import ComputeSELDResults
# from utils.accdoa import AccdoaResult, process_foa_input_accdoa_labels
# from utils.multi_accdoa import MultiAccdoaResult, process_foa_input_multi_accdoa_labels, MSELoss_ADPIT
# from utils.multi_accdoa import MultiAccdoaResult_2024
from utils.sed_doa import SedDoaDistResult, process_foa_input_sed_doa_labels, SedSCELoss


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def main(args):
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['result']['log_output_path'], filemode='w', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)
    data_process_fn = process_foa_input_sed_doa_labels
    result_class = SedDoaDistResult
    criterion = SedSCELoss(loss_weight=[1, 2])
    model = ResnetConformer_sed_coord_resnet50_medium_4_guided_m_1_weight_offline(in_channel=args['model']['in_channel'], in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])

    train_split = [3]
    train_dataset = LmdbDataset_logmel_resnet50(args['data']['train_lmdb_dir'], train_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['train_ignore'], segment_len=args['data']['segment_len'], data_process_fn=data_process_fn)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args['data']['batch_size'], shuffle=True, 
        num_workers=args['train']['train_num_workers'], collate_fn=train_dataset.collater
    )

    test_split = [4]
    test_dataset = LmdbDataset_logmel_resnet50(args['data']['test_lmdb_dir'], test_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['test_ignore'], segment_len=args['data']['segment_len'], data_process_fn=data_process_fn)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args['data']['batch_size'], shuffle=False, 
        num_workers=args['train']['test_num_workers'], collate_fn=test_dataset.collater
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    model = model.to(device)
    logger.info(model)
    set_random_seed(12332)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model']))
    logger.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args['train']['lr'])
    total_steps = args['train']['nb_steps']
    warmup_steps = int(total_steps*0.1)
    hold_steps = int(total_steps*0.6)
    decay_steps = int(total_steps*0.3)
    scheduler = TriStageLRScheduler(optimizer, peak_lr=args['train']['lr'], init_lr_scale=0.01, final_lr_scale=0.05, 
                                    warmup_steps=warmup_steps, hold_steps=hold_steps, decay_steps=decay_steps)
    epoch_count = 0
    step_count = 0

    stop_training = False
    while not stop_training:
        train_loss = []
        test_loss = []
        epoch_count += 1
        # train
        start_time = time.time()
        model.train()
        for data in train_dataloader:
            input = data['input'].to(device)
            v_input = data['av_feat'].to(device)
            target = data['target'].to(device)
            target[:, :, 52:] = target[:, :, 52:] * 0.01
            target[:, :, 13:26] = target[:, :, 13:26] * target[:, :, 52:]
            target[:, :, 26:39] = target[:, :, 26:39] * target[:, :, 52:]
            target[:, :, 39:52] = target[:, :, 39:52] * target[:, :, 52:]
            optimizer.zero_grad()
            output = model(input, v_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            step_count += 1
            if step_count % args['result']['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('epoch: {}, step: {}/{}, lr:{:.6f}, train_loss:{:.4f}'.format(epoch_count, step_count, total_steps, lr, loss.item()))
            if step_count == total_steps:
                stop_training = True
                break 
        torch.cuda.empty_cache()
        train_time = time.time() - start_time

        # test
        start_time = time.time()
        model.eval()
        test_result = result_class(segment_length=args['data']['segment_len'])
        for data in test_dataloader:
            input = data['input'].to(device)
            v_input = data['av_feat'].to(device)
            target = data['target'].to(device)
            target[:, :, 52:] = target[:, :, 52:] * 0.01
            target[:, :, 13:26] = target[:, :, 13:26] * target[:, :, 52:]
            target[:, :, 26:39] = target[:, :, 26:39] * target[:, :, 52:]
            target[:, :, 39:52] = target[:, :, 39:52] * target[:, :, 52:]
            with torch.no_grad():
                output = model(input, v_input)
                loss = criterion(output, target)
                test_loss.append(loss.item())

            output_dist = torch.sqrt(output[:,:,13:26]**2 + output[:,:,26:39]**2 + output[:,:,39:52]**2) 
            output = torch.cat((output, output_dist), dim=-1)
            test_result.add_items(data['wav_names'], output)
        output_dict = test_result.get_result()
        test_time = time.time() - start_time
        
        dcase_output_val_dir = os.path.join(args['result']['dcase_output_dir'], 'epoch{}_step{}'.format(epoch_count, step_count))
        os.makedirs(dcase_output_val_dir, exist_ok=True)
        for csv_name, perfile_out_dict in output_dict.items():
            output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
            write_output_format_file(output_file, perfile_out_dict)
        
        score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
        val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
        logger.info('epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_loss:{:.4f}, average_test_loss:{:.4f}'.format(epoch_count, step_count, total_steps, train_time, test_time, np.mean(train_loss), np.mean(test_loss)))
        logger.info('F/AE/Dist_err/Rel_dist_err/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr)))

        checkpoint_output_dir = args['result']['checkpoint_output_dir']
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_output_dir, 'checkpoint_epoch{}_step{}.pth'.format(epoch_count, step_count))
        torch.save(model.state_dict(), model_path)
        logger.info('save checkpoint: {}'.format(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args)