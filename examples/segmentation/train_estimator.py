"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
Author: Guocheng Qian @ 2022, guocheng.qian@kaust.edu.sa
"""
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample
from openpoints.models.layers import find_mps
from openpoints.models.layers import ball_query

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import math

import warnings

import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(gpu, cfg):
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    stride_list = cfg.model.encoder_args.strides
    stride = ''.join(str(e) for e in list(filter(lambda x: x != 1, stride_list)))

    # Batch size and loop must be 1 for fps extraction
    cfg.dataset.train.loop = 1
    train_loader = build_dataloader_from_cfg(1,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             precompute_fps=2,
                                             stride_list=cfg.model.encoder_args.strides,
                                             )
    
    train_train_loader = build_dataloader_from_cfg(1,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train-train',
                                             distributed=cfg.distributed,
                                             precompute_fps=2,
                                             stride_list=cfg.model.encoder_args.strides,
                                             )
    
    train_val_loader = build_dataloader_from_cfg(1,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train-val',
                                             distributed=cfg.distributed,
                                             precompute_fps=2,
                                             stride_list=cfg.model.encoder_args.strides,
                                             )

    val_loader = build_dataloader_from_cfg(1,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    
    training(train_train_loader, train_val_loader, val_loader, cfg) # Training
    inference(train_loader, val_loader, cfg) # Inference

class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def find_closest_indices(value, grid_points):
    for i in range(len(grid_points) - 1):
        if grid_points[i] <= value <= grid_points[i + 1]:
            return i, i + 1
    return len(grid_points) - 2, len(grid_points) - 1

def interpolate(value, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (value - x0) / (x1 - x0)

def linear_interpolation(outputs, target_positions, grid_points):
    interpolated_values = []

    for target in target_positions:
        lower_index, upper_index = find_closest_indices(target, grid_points)
        lower_value = grid_points[lower_index]
        upper_value = grid_points[upper_index]
        
        lower_output = outputs[lower_index]
        upper_output = outputs[upper_index]
        
        interpolated_value = interpolate(target, lower_value, upper_value, lower_output, upper_output)
        interpolated_values.append(interpolated_value)
    
    return interpolated_values

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
    
    def forward(self, output, target):
        mape = torch.abs((target - output) / (target))
        return torch.mean(mape)
    
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        loss = torch.sum(weights * (y_true - y_pred) ** 2) / torch.sum(weights)
        return loss
    
class PenaltyMSELoss(nn.Module): # Penalty Underestimated Results
    def __init__(self, penalty_factor=2.0):
        super(PenaltyMSELoss, self).__init__()
        self.penalty_factor = penalty_factor
    
    def forward(self, predictions, targets):
        diff = predictions - targets
        squared_diff = torch.square(diff)
        
        penalty = torch.where(diff < 0, squared_diff * self.penalty_factor, squared_diff)
        
        loss = torch.mean(penalty)
        return loss

def training(train_train_loader, train_val_loader, val_loader, cfg):
    # Choose the model to train
    model = SimpleNeuralNet(input_size=32, hidden_size1=128, hidden_size2=128, output_size=64).cuda()
    
    best_loss = float('inf')
    best_epoch = -1
    
    log_dir = 'mps_params/'
    param = 'mps_weight_S3DIS'
    best_epoch_log = os.path.join(log_dir, f'{param}.log')
    
    with open(best_epoch_log, 'w') as f:
        f.write("Epoch\tBest Epoch\tBest Loss\n")
    
    for epoch in range(20):
        strides = cfg.model.encoder_args.strides
        err_list = []
        
        # Re-initialize the train loader to shuffle the data each epoch
        train_train_loader = DataLoader(train_train_loader.dataset, batch_size=train_train_loader.batch_size, shuffle=True)
        pbar = tqdm(enumerate(train_train_loader), total=train_train_loader.__len__(), ascii=True)
        
        # Choose loss function
        # Option 1: Mean Squared Error Loss
        criterion = nn.MSELoss()
        
        # Option 2: Mean Absolute Percentage Error Loss
        # criterion = MAPELoss()
        
        # Option 3: Weighted Mean Squared Error Loss
        # criterion = WeightedMSELoss()
        
        # Option 4: Penalty MSE Loss (penalizes underestimation)
        # criterion = PenaltyMSELoss()
        
        # Choose optimizer
        # Option 1: Adam optimizer
        # optimizer = optim.Adam(model.parameters(), lr=0.025)
        
        # Option 2: Stochastic Gradient Descent
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        for idx, data in pbar:
            keys = data.keys() if callable(data.keys) else data.keys
            for key in keys:
                data[key] = data[key].cuda(non_blocking=True)
            xyz = data['pos']

            stride = 4
            num_sample = xyz.shape[1] // stride

            # Perform farthest point sampling to find the threshold
            mps = np.array(find_mps(xyz, num_sample).sqrt().reciprocal().squeeze(0).cpu())
            
            # Choose input and output positions for training
            # Option 1: 1/10 training
            input_positions = [1 / 10 * i / 32 for i in range(1, 33)]
            output_positions = [1 / 10 + 9 / 10 * i / 64 for i in range(1, 65)]
            
            # Interpolate the input and target data
            mps_grid_points = [(i + 1) / mps.shape[0] for i in range(mps.shape[0])]
            interpolated_input_data = linear_interpolation(mps, input_positions, mps_grid_points)
            interpolated_target_data = linear_interpolation(mps, output_positions, mps_grid_points)
            
            input_data = torch.tensor(interpolated_input_data, dtype=torch.float32).cuda()
            target_data = torch.tensor(interpolated_target_data, dtype=torch.float32).cuda()
            
            model.train()
            
            # Forward pass
            output = model(input_data)
            
            # Compute loss
            loss = criterion(output, target_data)
            
            # If using WeightedMSELoss, uncomment and use the following:
            '''
            weights = torch.tensor([(i + 1) / len(target_data) for i in range(len(target_data))]).cuda()
            loss = criterion(output, target_data, weights)
            '''
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            err_list.append(loss.item())
            pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Validation to check for best epoch
        model.eval()
        val_loss_list = []
        pbar = tqdm(enumerate(train_val_loader), total=train_val_loader.__len__(), ascii=True)
        with torch.no_grad():
            for idx, data in pbar:
                keys = data.keys() if callable(data.keys) else data.keys
                for key in keys:
                    data[key] = data[key].cuda(non_blocking=True)
                xyz = data['pos']
                
                stride = 4
                num_sample = xyz.shape[1] // stride
                
                mps = np.array(find_mps(xyz, num_sample).sqrt().reciprocal().squeeze(0).cpu())
                
                # Use the same input and output positions as in training
                # Option 1: 1/10 training
                input_positions = [1 / 10 * i / 32 for i in range(1, 33)]
                output_positions = [1 / 10 + 9 / 10 * i / 64 for i in range(1, 65)]
                
                mps_grid_points = [(i + 1) / mps.shape[0] for i in range(mps.shape[0])]
                interpolated_input_data = linear_interpolation(mps, input_positions, mps_grid_points)
                interpolated_target_data = linear_interpolation(mps, output_positions, mps_grid_points)
                
                input_data = torch.tensor(interpolated_input_data, dtype=torch.float32).cuda()
                target_data = torch.tensor(interpolated_target_data, dtype=torch.float32).cuda()
                
                # Forward pass
                output = model(input_data)
                
                # Compute validation loss
                val_loss = criterion(output, target_data)
                
                # If using WeightedMSELoss, uncomment and use the following:
                '''
                weights = torch.tensor([(i + 1) / len(target_data) for i in range(len(target_data))]).cuda()
                val_loss = criterion(output, target_data, weights)
                '''
                
                val_loss_list.append(val_loss.item())
                
                pbar.set_description(f"Epoch {epoch + 1}, Validation Loss: {val_loss.item():.4f}")

        print(f"Training Loss after Epoch {epoch + 1}: {np.mean(err_list):.4f}")
        torch.save(model.state_dict(), f'mps_params/{param}_epoch_{epoch + 1}.pth')
        
        mean_val_loss = np.mean(val_loss_list)
        print(f"Validation Loss after Epoch {epoch + 1}: {mean_val_loss:.4f}")
        
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'mps_params/{param}_best.pth')
            print(f"New Best Model found at Epoch {best_epoch}, Loss: {best_loss:.4f}")
        
        print(f"Best Epoch: {best_epoch}, Best Loss: {best_loss:.4f}")
        print()
        
        with open(best_epoch_log, 'a') as f:
            f.write(f"{epoch + 1}\t{best_epoch}\t{best_loss:.4f}\n")


def inference(train_loader, val_loader, cfg):
    model = SimpleNeuralNet(input_size=32, hidden_size1=128, hidden_size2=128, output_size=64).cuda()

    strides = cfg.model.encoder_args.strides
    fps_list = [None] * len(train_loader)
    err_ratio_lists = []

    # Inference
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), ascii=True)

    param = 'mps_weight_S3DIS_best.pth'
    model.load_state_dict(torch.load(f'mps_params/{param}'))
    model.eval()

    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        xyz = data['pos']

        stride = 4
        num_sample = xyz.shape[1] // stride

        # Farthest Point Sampling
        mps = np.array(find_mps(xyz, num_sample).sqrt().reciprocal().squeeze(0).cpu())

        # Define input and output positions
        input_positions = [1 / 10 * i / 32 for i in range(1, 33)]
        output_positions = [1 / 10 + 9 / 10 * i / 64 for i in range(1, 65)]

        # Interpolation
        mps_grid_points = [(i + 1) / mps.shape[0] for i in range(mps.shape[0])]
        interpolated_input_data = linear_interpolation(mps, input_positions, mps_grid_points)
        interpolated_target_data = linear_interpolation(mps, output_positions, mps_grid_points)

        input_data = torch.tensor(interpolated_input_data, dtype=torch.float32).cuda()
        target_data = torch.tensor(interpolated_target_data, dtype=torch.float32).cuda()

        # 1. MLP Inference
        with torch.no_grad():
            output = model(input_data)

        target_positions = [1 / 10 + 9 / 10 * (i + 1) / 64 for i in range(64)]
        interpolated_mps = linear_interpolation(mps, target_positions, mps_grid_points)

        differences_ratio = [(1 / output.cpu()[i].numpy() - 1 / interpolated_mps[i]) / (1 / interpolated_mps[i]) for i in range(64)]
        errors_ratio = [abs(diff) for diff in differences_ratio]
        err_ratio_lists.append(sum(errors_ratio) / len(errors_ratio))

    # Compute final metrics over all points
    mape_error = np.mean(err_ratio_lists) * 100
    print(f"MAPE: {mape_error}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, nargs='+', required=True, help='config file(s)')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()

    cfg1 = EasyConfig()
    cfg1.load(args.cfg[0], recursive=True)
    cfg1.update(opts)

    if cfg1.seed is None:
        cfg1.seed = np.random.randint(1, 10000)

    cfg1.rank, cfg1.world_size, cfg1.distributed, cfg1.mp = dist_utils.get_dist_info(cfg1)

    cfg1.task_name = args.cfg[0].split('.')[-2].split('/')[-2]
    cfg1.cfg_basename = args.cfg[0].split('.')[-2].split('/')[-1]
    tags = [
        cfg1.task_name,
        cfg1.mode,
        cfg1.cfg_basename,
        f'ngpus{cfg1.world_size}',
        f'seed{cfg1.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg1.root_dir = os.path.join(cfg1.root_dir, cfg1.task_name)

    cfg1.is_training = cfg1.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    main(0, cfg1)
