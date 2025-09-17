"""
PointMetaBase
"""
from re import I
from typing import List, Type
import math
import logging
import torch
import torch.nn as nn
import numpy as np
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, create_xyz_grouper, create_aggregation, furthest_point_sample, random_sample, three_interpolation, ball_query, find_mps, multi_level_filtering, fused_ball_query, fused_convert_ball_query, extract_ball_query, update_distance, remainder_fps, quick_fps, edgepc_sample
import copy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import gaussian_kde
import time

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))
from utils.cutils import grid_subsampling

def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


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

def linear_interpolation(outputs, target_positions, grid_points):
    indices = torch.searchsorted(grid_points, target_positions, right=False)

    l_idx = indices - 1
    r_idx = indices

    l_value = grid_points[l_idx]
    r_value = grid_points[r_idx]

    l_output = outputs[l_idx]
    r_output = outputs[r_idx]

    interpolated_values = l_output + (r_output - l_output) * (target_positions - l_value) / (r_value - l_value)
    return interpolated_values

def extract_by_index(outputs, target_positions, grid_points):
    indices = torch.searchsorted(grid_points, target_positions, right=True) - 1
    
    extracted_values = outputs[indices]
    return extracted_values

def multi_level_lfps(p, nsample, dataset, radius, model, qfps):
    # 1/10 FPS for estimation
    device = p.device
    
    if qfps:
        mps_reciprocal_list = quick_fps(p, nsample // 10)[:,3].sqrt().reciprocal().squeeze(0)
    else:
        mps_reciprocal_list = find_mps(p, nsample // 10).sqrt().reciprocal().squeeze(0)

    input_positions = torch.linspace(1 / 10 / 32, 1 / 10, steps=32, device=device)
    mps_grid_points = torch.linspace(1 / nsample, 1 / 10, steps=(nsample // 10), device=device)
    extracted_input_data = extract_by_index(mps_reciprocal_list, input_positions, mps_grid_points)
    input_data = torch.tensor(extracted_input_data, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        if 's3dis' in dataset.common.NAME.lower():
            output = model[0](input_data)
        elif 'scannet' in dataset.common.NAME.lower():
            output = model[1](input_data)
        elif 'semantickitti' in dataset.common.NAME.lower():
            output = model[2](input_data)
        else:
            print("Error: Dataset not specified for estimator!")
            exit(0)
    
    idx1 = torch.searchsorted(output, 1 / radius)
    rad1 = torch.tensor([1 / 10 + idx1 / len(output) * 9 / 10], device=device)
    
    grid_points = torch.linspace(1 / 10 + 1 / 64 * 9 / 10, 1, steps=64, device=device)
    '''
    # Exact Result
    mps_reciprocal_list = find_mps(p, nsample).sqrt().reciprocal().squeeze(0)[1:]
    output = mps_reciprocal_list
    
    idx1 = torch.searchsorted(output, 1 / radius)
    rad1 = torch.tensor([idx1 / len(output)], device=device)
    
    grid_points = torch.linspace(1 / len(output), 1, steps=len(output), device=device)
    '''
    
    #target_positions = torch.tensor([0,0,0,0,0,6/6], dtype=torch.float32, device=device)
    #target_positions = torch.tensor([0,0,0,0,3/6,6/6], dtype=torch.float32, device=device)
    #target_positions = torch.tensor([0,0,0,2/6,4/6,6/6], dtype=torch.float32, device=device)
    #target_positions = torch.tensor([0,0,3/12,6/12,9/12,12/12], dtype=torch.float32, device=device)
    #target_positions = torch.tensor([0,3/15,6/15,9/15,12/15,15/15], dtype=torch.float32, device=device)
    target_positions = torch.tensor([1/6, 2/6, 3/6, 4/6, 5/6, 6/6], dtype=torch.float32, device=device)
    rad1_idx = torch.searchsorted(target_positions, rad1).item()
    concat_positions = torch.cat((target_positions[:rad1_idx], rad1, target_positions[rad1_idx:]))
    
    mps = linear_interpolation(1 / output, concat_positions, grid_points)
    
    mps[rad1_idx] = radius # Adjust errors caused by interpolation
    if mps[rad1_idx] < mps[rad1_idx + 1]:
        mps[rad1_idx + 1], mps[rad1_idx] = mps[rad1_idx], mps[rad1_idx + 1]
        rad1_idx += 1
    elif mps[rad1_idx] > mps[rad1_idx - 1]:
        mps[rad1_idx - 1], mps[rad1_idx] = mps[rad1_idx], mps[rad1_idx - 1]
        rad1_idx -= 1
        
    mps = torch.flip(mps, dims=[0])
    
    rad1_idx = 6 - rad1_idx # flipped
    
    if 'semantickitti' in dataset.common.NAME.lower():
        filters = [128, 192, 256, 384, 512, 1024] # SemanticKITTI
        filters = filters[:rad1_idx] + [1024] + filters[rad1_idx:]
    else:
        filters = [16, 32, 48, 64, 80, 128]
        filters = filters[:rad1_idx] + [128] + filters[rad1_idx:]
    
    nfilter = torch.tensor(filters, device=device).int()

    fm = fused_ball_query(mps, nfilter, p, p).squeeze(0)    
    filter_matrix = torch.concat((fm[:, :sum(filters[:rad1_idx])].clone(), fm[:, sum(filters[:rad1_idx+1]):].clone()), dim = 1).contiguous()
    nfilter = torch.concat((nfilter[:rad1_idx],nfilter[rad1_idx+1:]))

    idx = multi_level_filtering(filter_matrix, nfilter, target_positions, nsample).long()
#    filters = [128, 192, 256, 384, 512, 1024] # SemanticKITTI
#    filters = [16, 32, 48, 64, 80, 128] # S3DIS and ScanNet
#    print(filter_matrix.shape[0])
#    print((filter_matrix[:, 0] - filter_matrix[:, filters[0] - 1]).count_nonzero())
#    print((filter_matrix[:, sum(filters[:1])] - filter_matrix[:, sum(filters[:2]) - 1]).count_nonzero())
#    print((filter_matrix[:, sum(filters[:2])] - filter_matrix[:, sum(filters[:3]) - 1]).count_nonzero())
#    print((filter_matrix[:, sum(filters[:3])] - filter_matrix[:, sum(filters[:4]) - 1]).count_nonzero())
#    print((filter_matrix[:, sum(filters[:4])] - filter_matrix[:, sum(filters[:5]) - 1]).count_nonzero())
#    print((filter_matrix[:, sum(filters[:5])] - filter_matrix[:, sum(filters[:6]) - 1]).count_nonzero())
#    exit(0)
    
    return idx, fm, rad1_idx

class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 fused_agg=False,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels1 = channels 
        convs1 = []
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels1) - 2) and not last_act else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)
        self.grouper = create_grouper(group_args)
        self.aggregation = create_aggregation(group_args)       # For fused aggregation
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type
        self.fused_agg = fused_agg

    def forward(self, pf, pe) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # preconv
        f = self.convs1(f)

        if self.fused_agg:
            # Fused Aggregation
            f = self.aggregation(p, p, pe, None, f)
        else:
            # grouping
            dp, fj = self.grouper(p, p, f)
            # pe + fj 
            f = pe + fj
            f = self.pool(f)

        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 fused_agg=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type
        self.fused_agg = fused_agg
        self.sampler = sampler

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels #if is_head else CHANNEL_MAP[feature_type](channels[0])
        channels1 = channels
        # channels2 = copy.copy(channels)
        channels2 = [in_channels] + [32,32] * (min(layers, 2) - 1) + [out_channels] # 16
        channels2[0] = 3
        convs1 = []
        convs2 = []

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args if not is_head else None,
                                             act_args=None if i == len(channels) - 2
                                                            and (self.use_res or is_head) else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)

        if not is_head:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                 norm_args=norm_args if not is_head else None,
                                                #  act_args=None if i == len(channels) - 2
                                                #                 and (self.use_res or is_head) else act_args,
                                                 act_args=act_args,
                                                **conv_args)
                            )
            self.convs2 = nn.Sequential(*convs2)

            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.xyz_grouper = create_xyz_grouper(group_args)
            self.aggregation = create_aggregation(group_args)       # For fused aggregation
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample
            elif sampler.lower() == 'lfps' or sampler.lower() == 'fused' or sampler.lower() == 'all':
                self.model_s3dis = SimpleNeuralNet(input_size=32, hidden_size1=128, hidden_size2=128, output_size=64).cuda()
                self.model_scannet = SimpleNeuralNet(input_size=32, hidden_size1=128, hidden_size2=128, output_size=64).cuda()
                self.model_semantickitti = SimpleNeuralNet(input_size=32, hidden_size1=128, hidden_size2=128, output_size=64).cuda()

                print("Loading estimator model weights...")
                self.model_s3dis.load_state_dict(torch.load('mps_params/32x128x128x64/S3DIS/mps_weight_S3DIS_best.pth'))
                self.model_scannet.load_state_dict(torch.load('mps_params/32x128x128x64/ScanNet/mps_weight_ScanNet_best.pth'))
                self.model_semantickitti.load_state_dict(torch.load('mps_params/32x128x128x64/SemanticKITTI/mps_weight_SemanticKITTI_best.pth'))
                
                self.model_s3dis.eval()
                self.model_scannet.eval()
                self.model_semantickitti.eval()

                self.model_list = [self.model_s3dis, self.model_scannet, self.model_semantickitti]
                
                self.radius = group_args.radius

                if sampler.lower() == 'all':
                    self.sample_fn = partial(multi_level_lfps, radius=self.radius, model=self.model_list, qfps=True)
                else:
                    self.sample_fn = partial(multi_level_lfps, radius=self.radius, model=self.model_list, qfps=False)
                    
            elif sampler.lower() == 'grid':
                self.sample_fn = grid_subsampling
            elif sampler.lower() == 'qfps':
                self.sample_fn = quick_fps
            elif sampler.lower() == 'edgepc':
                self.sample_fn = edgepc_sample

    def forward(self, pf_pe, dataset=None):
        p, f, pe = pf_pe
        self.ball_query_indices = []
        if self.is_head:
            f = self.convs1(f)  # (n, c)
        else:
            if not self.all_aggr:
                if self.sampler.lower() == 'lfps' or self.sampler.lower() == 'fused' or self.sampler.lower() == 'all':
                    if dataset is not None:
                        idx, filter_matrix, rad1_idx = self.sample_fn(p, p.shape[1] // self.stride, dataset)
                            
                        # Extract ball query result of radius r
                        if 'semantickitti' in dataset.common.NAME.lower():
                            filters = [128, 192, 256, 384, 512, 1024] # SemanticKITTI
                            filters = filters[:rad1_idx] + [1024] + filters[rad1_idx:]
                            ball_query = filter_matrix[:, sum(filters[:6]):sum(filters[:7])].unsqueeze(0).clone().contiguous()
                        else:
                            filters = [16, 32, 48, 64, 80, 128]
                            filters = filters[:rad1_idx] + [128] + filters[rad1_idx:]
                            ball_query = filter_matrix[:, sum(filters[:6]):sum(filters[:7])].unsqueeze(0).clone().contiguous()
                        
                        nvalid = (idx+1).count_nonzero()   # idx which failed to be sampled are marked as -1
                        #print(nvalid / (p.shape[1] // self.stride))
                        idx = idx[0][:nvalid].unsqueeze(0)
                        if nvalid != p.shape[1] // self.stride:
                            near_point_list = extract_ball_query(idx.int(), ball_query)    # Only the sampled points are extracted from the filter matrix
                            dist_matrix = update_distance(p, p[0][idx.squeeze(0)].clone().contiguous(), near_point_list)  # Update distance matrix so that FPS can be performed for the remaining iterations.
                            idx = remainder_fps(p, p.shape[1] // self.stride, idx, dist_matrix).long() # Those who failed to be sampled are sampled via FPS.

                        # Extract nns idx result of radius r
                        if self.sampler.lower() == 'fused' or self.sampler.lower() == 'all':
                            nns_idx = filter_matrix[idx.squeeze(0)][:, sum(filters[:rad1_idx]):sum(filters[:rad1_idx+1])][:, :32].unsqueeze(0).clone().contiguous()
                        else:
                            nns_idx, ball_query = None, None    
                    else:
                        idx = furthest_point_sample(p, p.shape[1] // self.stride).long()
                        nns_idx, ball_query = None, None
                elif self.sampler.lower() == 'grid':
                    if dataset is not None:
                        if 's3dis' in dataset.common.NAME.lower():
                            grid_size = 0.08
                        elif 'scannet' in dataset.common.NAME.lower():
                            grid_size = 0.04
                        elif 'semantickitti' in dataset.common.NAME.lower():
                            grid_size = 0.30
                        idx = self.sample_fn(p.squeeze(0).clone().cpu(), grid_size).unsqueeze(0).long().to(f.device)
                        nns_idx, ball_query = None, None
                    else:
                        idx = furthest_point_sample(p, p.shape[1] // self.stride).long()
                        nns_idx, ball_query = None, None
                elif self.sampler.lower() == 'qfps':
                    if dataset is not None:
                        idx = self.sample_fn(p.clone(), p.shape[1] // self.stride)[:, -1].long()
                    else:
                        idx = furthest_point_sample(p, p.shape[1] // self.stride).long()
                        # idx = self.sample_fn(p.clone(), p.shape[1] // self.stride).long()
                    nns_idx, ball_query = None, None
                elif self.sampler.lower() == 'edgepc' or self.sampler.lower() == 'afps':
                    if dataset is not None:
                        idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                    else:
                        idx = furthest_point_sample(p, p.shape[1] // self.stride).long()
                    nns_idx, ball_query = None, None
                else:
                    idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                    nns_idx, ball_query = None, None
 
                '''
                # Compare distance
                xyz = p.squeeze(0)
                rand_idx = random_sample(p, p.shape[1] // self.stride).squeeze(0).long()
                fps_idx = furthest_point_sample(p, p.shape[1] // self.stride).squeeze(0).long()
                lfps_idx = idx.squeeze(0).long()
                
                rand = xyz[rand_idx]
                fps = xyz[fps_idx]
                lfps = xyz[lfps_idx]
                
                rand_result = []
                fps_result = []
                lfps_result = []
                for i in tqdm(range(lfps.shape[0])):
                    rand_val = float((rand - xyz[rand_idx[i]]).square().sum(-1).sqrt().sort()[0][1])
                    fps_val = float((fps - xyz[fps_idx[i]]).square().sum(-1).sqrt().sort()[0][1])
                    lfps_val = float((lfps - xyz[lfps_idx[i]]).square().sum(-1).sqrt().sort()[0][1])
                    rand_result.append(rand_val)
                    fps_result.append(fps_val)
                    lfps_result.append(lfps_val)

                import matplotlib.pyplot as plt
                import seaborn as sns
                from scipy import stats
                rand_result = np.array(rand_result)
                fps_result = np.array(fps_result)
                lfps_result = np.array(lfps_result)
                print(fps_result)
                print(lfps_result)
                print(rand_result)
                fig = plt.subplots(figsize=(14,6.5))
                # sns.kdeplot(fps_result, color='g', shade=True, label='Furthest Point Sampling')
                # sns.kdeplot(lfps_result, color='b', shade=True, label='Lightweight FPS')
                # sns.kdeplot(rand_result, color='r', shade=True, label='Random Point Sampling')
                kde_fps = gaussian_kde(fps_result)
                kde_lfps = gaussian_kde(lfps_result)
                kde_rand = gaussian_kde(rand_result)
                
                if 'semantickitti' in dataset.common.NAME.lower():
                    x = np.linspace(0, 0.8, 1000)
                else:
                    x = np.linspace(0, 0.12, 1000)
                plt.plot(x, kde_fps(x), label='Furthest Point Sampling', color='g')
                plt.plot(x, kde_lfps(x), label='Lightweight FPS', color='b')
                plt.plot(x, kde_rand(x), label='Random Point Sampling', color='r')
                plt.fill_between(x, kde_fps(x), color='g', alpha=0.3)
                plt.fill_between(x, kde_lfps(x), color='b', alpha=0.3)
                plt.fill_between(x, kde_rand(x), color='r', alpha=0.3)
                # sns.histplot(fps_result, color='g', label='Furthest Point Sampling')
                # sns.histplot(lfps_result, color='b', label='Lightweight FPS')
                
                if 'semantickitti' in dataset.common.NAME.lower():
                    plt.xlim([0,0.8])
                else:
                    plt.xlim([0,0.12])
                
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('Minimum Point Spacing', fontsize=22)
                plt.ylabel('Probability', fontsize=22)
                plt.legend(fontsize=22)
                
                # plt.axvline(x=threshold, color='r', linestyle='--', linewidth=3)

                plt.savefig("profile.png")
                exit(0)
                '''
                
                if idx.dim() == 2:
                    new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                else:
                    new_p = torch.gather(p, 1, idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """

            """ DEBUG nfilter
            query_xyz, support_xyz = p, p
            radius1 = self.grouper.radius * 4
            radius2 = self.grouper.radius * 2
            radius3 = self.grouper.radius
            radius4 = self.grouper.radius * 0.5
            radius5 = self.grouper.radius * 0.25
            radius6 = self.grouper.radius * 0.125
            radius7 = self.grouper.radius * 0.0625
            dist = torch.cdist(query_xyz, support_xyz).cpu()
            points1 = len(dist[dist < radius1]) / (dist.shape[0] * dist.shape[1])
            points2 = len(dist[dist < radius2]) / (dist.shape[0] * dist.shape[1])
            points3 = len(dist[dist < radius3]) / (dist.shape[0] * dist.shape[1])
            points4 = len(dist[dist < radius4]) / (dist.shape[0] * dist.shape[1])
            points5 = len(dist[dist < radius5]) / (dist.shape[0] * dist.shape[1])
            points6 = len(dist[dist < radius6]) / (dist.shape[0] * dist.shape[1])
            points7 = len(dist[dist < radius7]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius1}, nfilter: {points1}')
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius2}, nfilter: {points2}')
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius3}, nfilter: {points3}')
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius4}, nfilter: {points4}')
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius5}, nfilter: {points5}')
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius6}, nfilter: {points6}')
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius7}, nfilter: {points7}')
            DEBUG end """

            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            # preconv
            f = self.convs1(f)

            if self.fused_agg:
                # Fused Aggregation
                # Must return idx! We want to perform ballquery only once...
                if nns_idx is None:
                    dp, nns_idx = self.xyz_grouper(new_p, p)
                else:
                    dp, _ = self.xyz_grouper(new_p, p, nns_idx)
                pe = self.convs2(dp)
                f = self.aggregation(new_p, p, pe, nns_idx, f)
                
                # Keep only the indices in ball_query that are present in idx for use in 3NN
                if ball_query is not None:
                    pos = fused_convert_ball_query(idx.int(), ball_query)
                    self.ball_query_indices.append(pos)
                else:
                    self.ball_query_indices.append(None)
            else:
                # grouping
                dp, fj = self.grouper(new_p, p, f)
                # conv on neighborhood_dp
                pe = self.convs2(dp)
                # pe + fj 
                f = pe + fj
                f = self.pool(f)
            
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f, pe


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None, idx=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2, idx)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2, idx))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 fused_agg=False,
                 num_posconvs=2,#2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args ,#if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args, fused_agg=fused_agg,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        elif num_posconvs == 4:
            channels = [in_channels, in_channels, in_channels, in_channels, in_channels]
        elif num_posconvs == 3:
            channels = [in_channels, in_channels, in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        identity = f
        f = self.convs([p, f], pe)
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f, pe]


@MODELS.register_module()
class PointMetaBaseEncoder(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        self.fused_agg = kwargs.get('fused_agg', False)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        pe_encoder = nn.ModuleList() #[]
        pe_grouper = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
            if i == 0:
                pe_encoder.append(nn.ModuleList())
                pe_grouper.append([])
            else:
                pe_encoder.append(self._make_pe_enc(
                    block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                    is_head=i == 0 and strides[i] == 1
                ))
                pe_grouper.append(create_grouper(group_args))
        self.encoder = nn.Sequential(*encoder)
        self.pe_encoder = pe_encoder #nn.Sequential(*pe_encoder)
        self.pe_grouper = pe_grouper
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_pe_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        ## for PE of this stage
        channels2 = [3, channels]
        convs2 = []
        if blocks > 1:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                norm_args=self.norm_args,
                                                act_args=self.act_args,
                                                **self.conv_args)
                            )
            convs2 = nn.Sequential(*convs2)
            return convs2
        else:
            return nn.ModuleList()

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, fused_agg=self.fused_agg, **self.aggr_args 
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res, fused_agg=self.fused_agg
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            pe = None
            p0, f0, pe = self.encoder[i]([p0, f0, pe])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None, dataset=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        idx_list = []
        for i in range(0, len(self.encoder)):
            if i == 0:
                pe = None
                _p, _f, _ = self.encoder[i]([p[-1], f[-1], pe])
            else:
                if i == 1:
                    _p, _f, _ = self.encoder[i][0]([p[-1], f[-1], pe], dataset)
                else:
                    _p, _f, _ = self.encoder[i][0]([p[-1], f[-1], pe])
                if self.blocks[i] > 1:
                    # grouping
                    dp, _ = self.pe_grouper[i](_p, _p, None)
                    # conv on neighborhood_dp
                    pe = self.pe_encoder[i](dp)
                    _p, _f, _ = self.encoder[i][1:]([_p, _f, pe])
            if hasattr(self.encoder[i][0], 'ball_query_indices'):
                idx_list.extend(self.encoder[i][0].ball_query_indices)
            p.append(_p)
            f.append(_f)
        return p, f, idx_list

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)

@MODELS.register_module()
class PointMetaBaseDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f, idx):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]], idx[i])])[1]
        return f[-len(self.decoder) - 1]
