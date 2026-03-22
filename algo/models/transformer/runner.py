from algo.models.transformer.data import TactileDataset, DataNormalizer
from torch.utils.data import DataLoader
from torch import optim
from algo.models.transformer.tact import MultiModalModel
from algo.models.models import AdaptTConv
from algo.models.transformer.utils import define_img_transforms, define_tactile_transforms, TactileTransform, \
    SyncCenterReshapeTransform, log_output
from algo.models.transformer.tcn import TCN

from tqdm import tqdm
import torch
import pickle
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import wandb
import time
from hydra.utils import to_absolute_path
from warmup_scheduler import GradualWarmupScheduler
from matplotlib import pyplot as plt
import cv2

import math
import torch.nn as nn

class DirectActionHead(nn.Module):
    """
    直接回归动作的 MLP 头。
    相比 FlowMatchingWrapper：
    - 推理时无需 ODE 积分，零额外开销
    - 输出确定性（同一观测 → 同一动作）
    - MSE loss 直接优化动作精度
    """
    def __init__(self, feature_extractor, feature_dim=256, action_dim=6):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.action_dim = action_dim
        
        self.action_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs_features):
        """直接从特征预测动作"""
        return self.action_head(obs_features)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [B, 1] representing time t in [0, 1]
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class FlowMatchingWrapper(nn.Module):
    """
    将基础特征提取器包裹成流匹配(Rectified Flow)模型
    """
    def __init__(self, feature_extractor, feature_dim=256, action_dim=6):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.action_dim = action_dim
        
        # 时间步 t 的 MLP 编码器
        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 2),
            nn.Mish(),
            nn.Linear(self.time_dim * 2, self.time_dim)
        )
        
        # 预测向量场(Vector Field)的动作头
        in_features = feature_dim + action_dim + self.time_dim
        self.action_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs_features, x_t, t):
        t_embed = self.time_mlp(t)
        inputs = torch.cat([obs_features, x_t, t_embed], dim=-1)
        return self.action_head(inputs)

class Runner:
    def __init__(self,
                 cfg=None,
                 agent=None,
                 action_regularization=False,
                 ):

        self.task_cfg = cfg
        self.cfg = cfg.offline_train
        self.agent = agent
        self.to_torch = lambda x: torch.from_numpy(x).float()
        self.only_bc = self.cfg.only_bc
        self.ppo_step = agent.play_latent_step if ((agent is not None) and (action_regularization)) else None
        self.optimizer = None
        self.scheduler = None
        self.tact = None
        self.sequence_length = self.cfg.model.transformer.sequence_length

        # Device management
        if torch.cuda.is_available() and self.cfg.multi_gpu:
            torch.multiprocessing.set_start_method("spawn", force=True)

            available_gpus = list(range(torch.cuda.device_count()))
            print("Available GPU IDs:", available_gpus)

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Check if gpu_ids are valid and meet the memory requirement
            valid_gpu_ids = []
            for gpu_id in self.cfg.gpu_ids:
                if gpu_id in available_gpus:
                    memory_usage = get_gpu_memory_usage(gpu_id)
                    if memory_usage <= 0.25:
                        valid_gpu_ids.append(gpu_id)
            print(valid_gpu_ids)
            if not valid_gpu_ids:
                print("No valid gpu. Exit!")
                exit()
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in valid_gpu_ids])
                print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])

                first_gpu_id = valid_gpu_ids[0] if valid_gpu_ids else 0
                device = torch.device(f"cuda:{first_gpu_id}")
        else:
            device = f'cuda:{self.cfg.gpu_ids[0]}'

        self.device = device

        self._init_transforms()

        self.init_model()

    def init_model(self):
        out_size = self.cfg.model.transformer.output_size
        out_size = 6 if self.only_bc else out_size
        
        # --- 新增：所有视觉/触觉基础网络只输出 256 维特征 ---
        feature_dim = 256 
        
        if self.cfg.model.model_type == 'tact':
            add_lin = self.cfg.model.tact.output_size if self.cfg.model.transformer.load_tact else 0
            pcl_conf = {'num_sample_plug':  self.task_cfg.task.env.num_points,
                        'num_sample_hole': self.task_cfg.task.env.num_points_socket,
                        'num_sample_goal': self.task_cfg.task.env.num_points_goal,
                        'num_sample_all': self.task_cfg.task.env.num_points_goal,
                        'merge_socket':  self.task_cfg.task.env.merge_socket_pcl,
                        'merge_goal':  self.task_cfg.task.env.merge_goal_pcl,
                        'scene_pcl': self.task_cfg.task.env.include_all_pcl,
                        'merge_plug': self.task_cfg.task.env.include_plug_pcl,
                        'relative': False}

            base_model = MultiModalModel(context_size=self.sequence_length,
                                         num_channels=self.tactile_channel,
                                         num_lin_features=self.cfg.model.linear.input_size,
                                         num_outputs=feature_dim, # 修改处
                                         tactile_encoder="depth",
                                         img_encoder="depth",
                                         seg_encoder="depth",
                                         lin_encoding_size=self.cfg.model.transformer.lin_encoding_size,
                                         tactile_encoding_size=self.cfg.model.transformer.tactile_encoding_size,
                                         img_encoding_size=self.cfg.model.transformer.img_encoding_size,
                                         seg_encoding_size=self.cfg.model.transformer.seg_encoding_size,
                                         mha_num_attention_heads=self.cfg.model.transformer.num_heads,
                                         mha_num_attention_layers=self.cfg.model.transformer.num_layers,
                                         mha_ff_dim_factor=self.cfg.model.transformer.dim_factor,
                                         additional_lin=add_lin,
                                         include_img=self.cfg.model.use_img,
                                         include_seg=self.cfg.model.use_seg,
                                         include_lin=self.cfg.model.use_lin,
                                         include_pcl=self.cfg.model.use_pcl,
                                         include_tactile=self.cfg.model.use_tactile,
                                         only_bc=self.only_bc, pcl_conf=pcl_conf)

        elif self.cfg.model.model_type == 'simple':
            base_model = AdaptTConv(ft_dim=self.cfg.model.linear.input_size, ft_out_dim=feature_dim)
        elif self.cfg.model.model_type == 'tcn':
            base_model = TCN(input_size=self.cfg.model.linear.input_size,
                             output_size=feature_dim, num_channels=[128] * 3, kernel_size=5, dropout=0).to(self.device)
        else:
            assert NotImplementedError

        # === 关键修改：根据 only_bc 选择不同的动作头 ===
        use_flow_matching = getattr(self.cfg, 'use_flow_matching', False)
    
        if use_flow_matching:
            # 多模态场景：使用 FlowMatchingWrapper
            self.model = FlowMatchingWrapper(base_model, feature_dim=feature_dim, action_dim=out_size)
            self._use_flow = True
            print("[Model] Using FlowMatchingWrapper (ODE-based)")
        else:
            # 确定性场景（推荐）：直接回归
            self.model = DirectActionHead(base_model, feature_dim=feature_dim, action_dim=out_size)
            self._use_flow = False
            print("[Model] Using DirectActionHead (deterministic)")
    
        self.model.to(self.device)
        return self.model

    def _init_transforms(self):

        # img
        self.img_channel = 1 if self.cfg.img_type == "depth" else 3
        self.img_color_jitter = self.cfg.img_color_jitter
        self.img_width = self.cfg.img_width
        self.img_height = self.cfg.img_height
        self.crop_img_width = self.img_width - self.cfg.img_crop_w
        self.crop_img_height = self.img_height - self.cfg.img_crop_h
        self.img_transform, self.seg_transform, self.img_eval_transform, self.sync_transform, self.sync_eval_transform = define_img_transforms(
            self.img_width,
            self.img_height,
            self.crop_img_width,
            self.crop_img_height,
            self.cfg.img_patch_size,
            self.cfg.img_gaussian_noise,
            self.cfg.img_masking_prob
        )

        self.sync_eval_reshape_transform = SyncCenterReshapeTransform((self.crop_img_width,
                                                                       self.crop_img_height),
                                                                      self.img_eval_transform,
                                                                      self.img_eval_transform)

        # tactile
        self.num_fingers = 3
        self.tactile_channel = 1 if self.cfg.tactile_type == "gray" else 3
        self.tactile_color_jitter = self.cfg.tactile_color_jitter
        self.tactile_width = self.cfg.tactile_width
        self.tactile_height = self.cfg.tactile_height
        self.crop_tactile_width = self.tactile_width - self.cfg.tactile_crop_w
        self.crop_tactile_height = self.tactile_height - self.cfg.tactile_crop_h
        self.tactile_transform, self.tactile_eval_transform = define_tactile_transforms(
            self.tactile_width,
            self.tactile_height,
            self.crop_tactile_width,
            self.crop_tactile_height,
            self.cfg.tactile_patch_size,
            self.cfg.tactile_gaussian_noise,
            self.cfg.tactile_masking_prob
        )
        self.process_tactile = TactileTransform(self.tactile_transform)
        self.eval_process_tactile = TactileTransform(self.tactile_eval_transform)

    def train(self, dl, val_dl, ckpt_path, print_every=50, eval_every=250, test_every=500):
        """
        Train the model using the provided data loader, with periodic validation and testing.

        Args:
            dl (DataLoader): Training data loader.
            val_dl (DataLoader): Validation data loader.
            ckpt_path (str): Path to save checkpoints.
            print_every (int): Frequency of printing training loss.
            eval_every (int): Frequency of evaluating the model on validation data.
            test_every (int): Frequency of testing the model.
        """
        self.model.train()
        train_loss, val_loss = [], []
        latent_loss_list, action_loss_list = [], []
        d_pos_rpy = None
        progress_bar = tqdm(enumerate(dl), total=len(dl), desc="Training Progress", unit="batch")

        for i, (tactile, img, seg, stud_obs, pos_rpy, obs_hist, latent, action) in progress_bar:

            self.model.train()

            tactile = tactile.to(self.device)  # [B T F W H C]
            img = img.to(self.device)
            seg = seg.to(self.device)
            stud_obs = stud_obs.to(self.device)
            latent = latent.to(self.device)
            action = action.to(self.device)

            if self.tact is not None:
                assert NotImplementedError
                d_pos_rpy = self.tact(tactile, img, seg, stud_obs).unsqueeze(1)

            if self.only_bc:
                out = self.model(tactile, img, seg, stud_obs, add_lin_input=d_pos_rpy)
                # out = torch.clamp(out, -1, 1)
                loss_latent = self.loss_fn_mean(out, action[:, -1, :])
            else:
                out = self.model(tactile, img, seg, stud_obs, add_lin_input=d_pos_rpy)
                loss_latent = self.loss_fn_mean(out, latent[:, -1, :])

            loss_action = torch.zeros(1, device=self.device)

            if self.ppo_step is not None:
                obs_hist = obs_hist[:, -1, :].to(self.device).view(obs_hist.shape[0], obs_hist.shape[-1])
                pred_action, _ = self.ppo_step({'obs': obs_hist, 'latent': out})
                # pred_action = torch.clamp(pred_action, -1, 1)
                loss_action = self.loss_fn_mean(pred_action, action[:, -1, :])

            loss = (self.cfg.train.latent_scale * loss_latent) + (self.cfg.train.action_scale * loss_action)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            train_loss.append(loss.item())
            latent_loss_list.append(loss_latent.item())
            if self.ppo_step is not None:
                action_loss_list.append(loss_action.item())

            # Update tqdm description
            progress_bar.set_postfix({
                'Batch': i + 1,
                'Loss': np.mean(train_loss)
            })

            if (i + 1) % print_every == 0 or (i == len(dl) - 1):
                print(f'step {i + 1}:', np.mean(train_loss))
                self._wandb_log({'train/loss': np.mean(train_loss),
                                 'train/latent_loss': np.mean(latent_loss_list)})
                if self.ppo_step is not None:
                    self._wandb_log({'train/action_loss': np.mean(action_loss_list)})

                self.train_loss.append(np.mean(train_loss))
                self.ax1.plot(self.train_loss, '-ro', linewidth=3, label='train loss')
                train_loss = []
                latent_loss_list = []
                action_loss_list = []

            if (i + 1) % eval_every == 0 or (i == len(dl) - 1):
                val_loss = self.validate(val_dl)
                print(f'validation loss: {val_loss}')
                self.val_loss.append(val_loss)
                self.ax1.plot(self.val_loss, '-ko', linewidth=3, label='val loss')
                # self.ax1.legend()
                self.fig.savefig(f'{self.save_folder}/train_val_comp.png', dpi=200, bbox_inches='tight')
                # self.fig.clf()

                log_output(tactile,
                           img,
                           seg,
                           stud_obs,
                           out,
                           action if self.only_bc else latent,
                           pos_rpy,
                           self.save_folder,
                           d_pos_rpy,
                           'train',
                           )

                self.model.train()

            # if (i + 1) % test_every == 0:
            # try:
            #     self.test()
            # except Exception as e:
            #     print(f'Error during test: {e}')
            # self.model.train()

        return val_loss

    def validate(self, dl):
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            latent_loss_list, action_loss_list = [], []
            d_pos_rpy = None
            for i, (tactile, img, seg, stud_obs, pos_rpy, obs_hist, latent, action) in tqdm(enumerate(dl)):

                tactile = tactile.to(self.device)
                img = img.to(self.device)
                seg = seg.to(self.device)

                stud_obs = stud_obs.to(self.device)
                latent = latent.to(self.device)
                action = action.to(self.device)

                if self.tact is not None:
                    assert NotImplementedError
                    d_pos_rpy = self.tact(tactile, img, seg, stud_obs).unsqueeze(1)

                if self.only_bc:
                    out = self.model(tactile, img, seg, stud_obs, add_lin_input=d_pos_rpy)
                    out = torch.clamp(out, -1, 1)
                    loss_latent = self.loss_fn_mean(out, action[:, -1, :])
                else:
                    out = self.model(tactile, img, seg, stud_obs, add_lin_input=d_pos_rpy)
                    loss_latent = self.loss_fn_mean(out, latent[:, -1, :])

                loss_action = torch.zeros(1, device=self.device)

                if self.ppo_step is not None:
                    obs_hist = obs_hist[:, -1, :].to(self.device).view(obs_hist.shape[0], obs_hist.shape[-1])
                    pred_action, _ = self.ppo_step({'obs': obs_hist, 'latent': out})
                    pred_action = torch.clamp(pred_action, -1, 1)
                    loss_action = self.loss_fn_mean(pred_action, action[:, -1, :])

                # TODO: add scaling loss coefficients
                loss = (self.cfg.train.latent_scale * loss_latent) + (self.cfg.train.action_scale * loss_action)

                val_loss.append(loss.item())
                latent_loss_list.append(loss_latent.item())
                if self.ppo_step is not None:
                    action_loss_list.append(loss_action.item())

            # self._wandb_log({
            #     'val/loss': np.mean(val_loss),
            #     'val/latent_loss': np.mean(latent_loss_list),
            # 'val/action_loss': np.mean(action_loss_list)
            # })
            # if self.ppo_step is not None:
            #     self._wandb_log({
            #         'val/action_loss': np.mean(action_loss_list)
            #     })

            log_output(tactile,
                       img,
                       seg,
                       stud_obs,
                       out,
                       action if self.only_bc else latent,
                       pos_rpy,
                       self.save_folder,
                       d_pos_rpy,
                       'valid',
                       )

        return np.mean(val_loss)

    def predict(self, obs_dict, requires_grad=False, display=False, num_inference_steps=10, ablate_tactile=False):
        if not requires_grad:
            self.model.eval()
            with torch.no_grad():
                return self._predict_forward(obs_dict, display, num_inference_steps, ablate_tactile)
        else:
            return self._predict_forward(obs_dict, display, num_inference_steps, ablate_tactile)

    def _extract_features(self, obs_dict, display=False, ablate_tactile=False):
        """只执行一次沉重的 CNN/Transformer 运算"""
        tactile = obs_dict['tactile'] if 'tactile' in obs_dict else None
        img = obs_dict['img'] if 'img' in obs_dict else None
        seg = obs_dict['seg'] if 'seg' in obs_dict else None
        student_obs = obs_dict['student_obs'] if 'student_obs' in obs_dict else None
        pcl = obs_dict['pcl'] if 'pcl' in obs_dict else None

        if self.cfg.model.use_tactile:
            tactile = tactile.to(self.device)
            # ===== [核心修改：触觉消融开关] =====
            if ablate_tactile:
                tactile = torch.zeros_like(tactile)
            # ====================================
            if self.tactile_transform is not None:
                if tactile.ndim == 4:
                    tactile = tactile.reshape(*tactile.shape[:2], self.num_fingers, 1, self.crop_tactile_width, self.crop_tactile_height)
                tactile = self.eval_process_tactile(tactile).to(self.device)
        if self.cfg.model.use_img:
            img, seg = img.to(self.device), seg.to(self.device)
            if img.ndim == 3:
                img = img.reshape(*img.shape[:2], 1, self.crop_img_width, self.crop_img_height)
                seg = seg.reshape(*seg.shape[:2], 1, self.crop_img_width, self.crop_img_height)
            img, seg = self.sync_eval_reshape_transform(img, seg)
        if self.cfg.model.use_lin: student_obs = student_obs.to(self.device)
        if self.cfg.model.use_pcl: pcl = pcl.to(self.device)

        if self.tact is not None:
            d_pos_rpy = self.tact(tactile, img, student_obs).unsqueeze(1)
            # 注意调用的是 model.feature_extractor
            out = self.model.feature_extractor(tactile, img, student_obs, add_lin_input=d_pos_rpy)
        else:
            out = self.model.feature_extractor(obs_tactile=tactile, obs_img=img, obs_seg=seg, lin_input=student_obs, obs_pcl=pcl)
        return out

    def _predict_forward(self, obs_dict, display=False, num_inference_steps=10, ablate_tactile=False):
        obs_features = self._extract_features(obs_dict, display, ablate_tactile)
    
        if self._use_flow:
            # Flow matching: ODE 积分
            B = obs_features.shape[0]
            x = torch.randn((B, self.model.action_dim), device=self.device)
            dt = 1.0 / num_inference_steps
            for i in range(num_inference_steps):
                t_val = i * dt
                t_tensor = torch.full((B, 1), t_val, device=self.device)
                v_pred = self.model(obs_features, x, t_tensor)
                x = x + v_pred * dt
            return x, None
        else:
            # 直接回归: 一次前向传播
            actions = self.model(obs_features)
            return actions, None

    def compute_flow_loss(self, student_dict, x_1, weights):
        """
        根据模型类型自动切换 loss 计算方式：
        - FlowMatchingWrapper: 原有 rectified flow loss
        - DirectActionHead:    加权 MSE loss
        """
        obs_features = self._extract_features(student_dict, display=False)
    
        if self._use_flow:
            # 原有 flow matching loss
            B = x_1.shape[0]
            device = self.device
            x_0 = torch.randn_like(x_1)
            t = torch.rand((B, 1), device=device)
            x_t = t * x_1 + (1.0 - t) * x_0
            v_target = x_1 - x_0
            v_pred = self.model(obs_features, x_t, t)
            loss_elem = (v_pred - v_target) ** 2
            weighted_loss = loss_elem * weights.unsqueeze(0)
            return weighted_loss.mean()
        else:
            # DirectActionHead: 使用加权 SmoothL1 / Huber
            # 大误差阶段比 MSE 稳，小误差阶段又比纯 L1 更利于精修
            pred_actions = self.model(obs_features)
            diff = pred_actions - x_1

            beta = 0.05
            abs_diff = diff.abs()
            huber = torch.where(
                abs_diff < beta,
                0.5 * (diff ** 2) / beta,
                abs_diff - 0.5 * beta
            )

            weighted_loss = huber * weights.unsqueeze(0)
            return weighted_loss.mean()

    def test(self):

        with torch.inference_mode():
            stats = self.stats.copy()
            num_success, total_trials = self.agent.test(self.predict, stats)
            if total_trials > 0:
                print(f'{num_success}/{total_trials}, success rate on :', num_success / total_trials)
                self._wandb_log({
                    'test/success_rate': num_success / total_trials
                })
            else:
                print('something went wrong, there are no test trials')

    def load_model(self, model_path, device='cuda:0'):
        print('Loading Multimodal model:', model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        # self.model.eval()
        self.device = device
        self.model.to(device)

    def load_tact_model(self, tact_path, device='cuda:0'):
        print('Loading Tactile model:', tact_path)
        self.tact.load_state_dict(torch.load(tact_path))
        # self.model.eval()
        self.device = device
        self.tact.to(device)

    def run_train(self, file_list, save_folder, epochs=100, train_test_split=0.9, train_batch_size=32,
                  val_batch_size=32, learning_rate=1e-4, device='cuda:0', print_every=50, eval_every=250,
                  test_every=500):

        random.shuffle(file_list)
        print('# trajectories:', len(file_list))

        ckpt_path = f'{save_folder}/checkpoints'
        if not os.path.exists(ckpt_path):
            os.makedirs(f'{ckpt_path}')

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)

        if self.cfg.train.scheduler == 'reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode='min',
                                                                  factor=0.5,
                                                                  patience=3,
                                                                  verbose=True)

        if self.cfg.train.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        if self.cfg.train.warmup:
            print("Using warmup scheduler")
            self.scheduler = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1,
                total_epoch=self.cfg.train.warmup_epochs,
                after_scheduler=self.scheduler,
            )

        num_train_envs = int(len(file_list) * train_test_split)
        train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
        val_idxs = np.arange(num_train_envs, len(file_list)).astype(int).tolist()
        training_files = [file_list[i] for i in train_idxs]
        val_files = [file_list[i] for i in val_idxs]

        # Passing trajectories
        train_ds = TactileDataset(traj_files=training_files,
                                  sequence_length=self.sequence_length,
                                  stats=self.stats,
                                  img_transform=self.img_transform,
                                  seg_transform=self.seg_transform,
                                  sync_transform=self.sync_transform,
                                  tactile_transform=self.tactile_transform,
                                  include_img=self.cfg.model.use_img,
                                  include_seg=self.cfg.model.use_seg,
                                  include_lin=self.cfg.model.use_lin,
                                  include_tactile=self.cfg.model.use_tactile,
                                  obs_keys=self.cfg.train.obs_keys,
                                  )

        train_dl = DataLoader(train_ds,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              persistent_workers=True,
                              )

        val_ds = TactileDataset(traj_files=val_files,
                                sequence_length=self.sequence_length,
                                stats=self.stats,
                                img_transform=self.img_eval_transform,
                                seg_transform=self.img_eval_transform,
                                sync_transform=self.sync_eval_transform,
                                tactile_transform=self.tactile_eval_transform,
                                include_img=self.cfg.model.use_img,
                                include_seg=self.cfg.model.use_seg,
                                include_lin=self.cfg.model.use_lin,
                                include_tactile=self.cfg.model.use_tactile,
                                obs_keys=self.cfg.train.obs_keys,
                                )

        val_dl = DataLoader(val_ds,
                            batch_size=val_batch_size,
                            shuffle=True,
                            num_workers=16,
                            pin_memory=True,
                            persistent_workers=True,
                            )

        # training
        for epoch in range(epochs):
            self.validate(val_dl)

            if self.cfg.train.only_validate:
                self.validate(val_dl)
            else:
                val_loss = self.train(train_dl, val_dl, ckpt_path,
                                      print_every=print_every,
                                      eval_every=eval_every,
                                      test_every=test_every)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(np.mean(val_loss))
                    else:
                        self.scheduler.step()
                print('Saving the model')
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                torch.save(raw_model.state_dict(), f'{ckpt_path}/model_last.pt')  # {epoch}.pt')

    def _wandb_log(self, data):
        if self.cfg.wandb.wandb_enabled:
            wandb.log(data)

    def run(self):

        self.loss_fn_mean = torch.nn.MSELoss(reduction='mean')
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        self.fig = plt.figure(figsize=(20, 15))
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.train_loss, self.val_loss = [], []

        # Load student checkpoint.
        if self.cfg.train.load_checkpoint:
            model_path = self.cfg.train.student_ckpt_path
            self.load_model(model_path, device=self.device)

        if self.cfg.model.transformer.load_tact:
            tact_path = self.cfg.model.transformer.tact_path
            self.load_tact_model(tact_path, device=self.device)

        train_config = {
            "epochs": self.cfg.train.epochs,
            "train_test_split": self.cfg.train.train_test_split,
            "train_batch_size": self.cfg.train.train_batch_size,
            "val_batch_size": self.cfg.train.val_batch_size,
            "learning_rate": self.cfg.train.learning_rate,
            "print_every": self.cfg.train.print_every,
            "eval_every": self.cfg.train.eval_every,
            "test_every": self.cfg.train.test_every
        }

        from datetime import datetime
        from glob import glob

        print('Loading trajectories from', self.cfg.data_folder)

        file_list = glob(os.path.join(self.cfg.data_folder, '*/*/obs/*.npz'))
        save_folder = f'{to_absolute_path(self.cfg.output_dir)}/{self.cfg.model.model_type}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder

        # Either create new norm or load existing
        normalizer = DataNormalizer(self.cfg, file_list, self.cfg.data_folder)
        normalizer.run()
        self.stats = normalizer.stats

        if self.cfg.train.only_test:
            print('Only testing')
            self.test()

        if self.cfg.wandb.wandb_enabled:
            wandb.init(
                # Set the project where this run will be logged
                project=self.cfg.wandb.wandb_project_name,
                # Track hyperparameters and run metadata
                config=train_config,
                dir=save_folder,
            )

        with open(os.path.join(save_folder, f"task_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.task_cfg))
        with open(os.path.join(save_folder, f"train_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if self.cfg.multi_gpu and len(self.cfg.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.cfg.gpu_ids)

        self.model = self.model.to(self.device)

        self.run_train(file_list, save_folder, device=self.device, **train_config)


def get_gpu_memory_usage(device_id):
    """
    Returns the memory usage of the specified GPU in terms of total memory
    and allocated memory as a percentage.
    """
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    allocated_memory = torch.cuda.memory_allocated(device_id)
    return allocated_memory / total_memory
