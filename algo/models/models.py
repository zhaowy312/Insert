# --------------------------------------------------------
# ReDi-LPD Student Architecture & Standard ActorCritic
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# === [ReDi-LPD] Dependencies ===
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from algo.models.diffusion_transformer import DiffusionTransformer
# ===============================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(layer_init(nn.Linear(input_size, output_size)))
            layers.append(nn.Tanh())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AdaptTConv(nn.Module):
    def __init__(self, ft_dim=6 * 5, ft_out_dim=32):
        super(AdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(ft_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, ft_out_dim)
        self.ft_dim = ft_dim
    def forward(self, tac, img, seg, x):

        x = x.reshape(x.shape[0], 30, self.ft_dim)
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)

        actions_num = kwargs['actions_num']
        input_shape = kwargs['input_shape']
        mlp_input_shape = input_shape[0]
        self.units = kwargs['actor_units']
        out_size = self.units[-1]
        self.ft_info = kwargs["ft_info"]
        self.tactile_info = kwargs["tactile_info"]
        self.obs_info = kwargs["obs_info"]
        self.contact_info = kwargs['gt_contacts_info']
        self.contact_mlp_units = kwargs['contacts_mlp_units']
        self.only_contact = kwargs['only_contact']
        self.priv_mlp_units = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['extrin_adapt']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.shared_parameters = kwargs['shared_parameters']

        self.temp_latent = []
        self.temp_extrin = []

        if self.priv_info:
            mlp_input_shape += self.priv_mlp_units[-1]

            if self.contact_info:
                self.priv_info_dim += self.contact_mlp_units[-1]
                self.contact_mlp = MLP(units=self.contact_mlp_units, input_size=kwargs["num_contact_points"])

            self.env_mlp = MLP(units=self.priv_mlp_units, input_size=self.priv_info_dim)

            if self.priv_info_stage2:
                # ---- tactile Decoder ----
                # Dims of latent have to be the same |z_t - z'_t|
                if self.obs_info:
                    self.obs_units = kwargs["obs_units"]
                    self.obs_mlp = MLP(
                        units=self.obs_units, input_size=kwargs["student_obs_input_shape"])

                if self.tactile_info:
                    path_checkpoint, root_dir = None, None
                    if False:  # kwargs['tactile_pretrained']
                        # we should think about decoupling this problem and pretraining a decoder
                        import os
                        current_file = os.path.abspath(__file__)
                        root_dir = os.path.abspath(
                            os.path.join(current_file, "..", "..", "..", "..", "..")
                        )
                        path_checkpoint = kwargs["checkpoint_tactile"]

                    # load a simple tactile decoder
                    tactile_decoder_embed_dim = kwargs['tactile_decoder_embed_dim']
                    tactile_input_dim = kwargs['tactile_input_dim']
                    num_channels = tactile_input_dim[-1]
                    num_fingers = 3

                    # self.tactile_decoder_m = load_tactile_resnet(tactile_decoder_embed_dim, 3 * num_channels)
                    self.tactile_decoder = load_tactile_resnet(tactile_decoder_embed_dim // num_fingers, num_channels)

                    # add tactile mlp to the decoded features
                    self.tactile_units = kwargs["mlp_tactile_units"]
                    if not self.obs_info and self.tactile_info:
                        self.tactile_units[-1] = self.priv_mlp_units[-1]  # self.tactile_units[-1] // 2
                    tactile_input_shape = kwargs["mlp_tactile_input_shape"]

                    self.tactile_mlp = MLP(
                        units=self.tactile_units, input_size=tactile_input_shape
                    )

                if self.obs_info and self.tactile_info:
                    self.merge_units = kwargs["merge_units"]
                    self.merge_mlp = MLP(
                        units=self.merge_units, input_size=self.tactile_units[-1] + self.obs_units[-1]
                    )

                if self.ft_info:
                    assert 'ft is not supported yet, force rendering is currently ambiguous'
                    self.ft_units = kwargs["ft_units"]
                    ft_input_shape = kwargs["ft_input_shape"]
                    self.ft_adapt_tconv = FTAdaptTConv(ft_dim=ft_input_shape,
                                                       ft_out_dim=self.ft_units[-1])

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        if not self.shared_parameters:
            self.critic_mlp = MLP(units=self.units, input_size=mlp_input_shape)

        self.value = layer_init(torch.nn.Linear(out_size, 1), std=1.0)
        self.mu = layer_init(torch.nn.Linear(out_size, actions_num), std=0.01)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        self.fig = plt.figure(figsize=(8, 6))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),  # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, latent, _ = self._actor_critic(obs_dict)
        return mu, latent

    def _actor_critic(self, obs_dict, display=False):

        obs = obs_dict['obs']
        extrin, extrin_gt = None, None

        # Transformer student ( latent pass is in frozen_ppo)
        if 'latent' in obs_dict and obs_dict['latent'] is not None:
            extrin = obs_dict['latent']
            obs = torch.cat([obs, extrin], dim=-1)

            if 'priv_info' in obs_dict:
                extrin_gt = self.env_mlp(obs_dict['priv_info'])
                # extrin_gt = torch.tanh(extrin_gt)

                if display:
                    plt.ylim(-1, 1)
                    plt.scatter(list(range(extrin_gt.shape[-1])), extrin.clone().detach().cpu().numpy()[0, :],
                                color='r')
                    plt.scatter(list(range(extrin_gt.shape[-1])), extrin_gt.clone().cpu().numpy()[0, :], color='b')
                    plt.pause(0.0001)
                    plt.cla()

        # MLP models
        else:
            # Contact obs with extrin/gt_extrin and pass to the actor
            if self.priv_info:
                if self.priv_info_stage2:

                    if self.tactile_info:
                        extrin_tactile = self._tactile_encode_multi(obs_dict['tactile_hist'])
                    if self.obs_info:
                        extrin_obs = self.obs_mlp(obs_dict['student_obs'])
                    # If both, merge and create student extrin
                    if self.obs_info and self.tactile_info:
                        extrin = torch.cat([extrin_tactile, extrin_obs], dim=-1)
                        extrin = self.merge_mlp(extrin)
                    elif self.tactile_info:
                        extrin = extrin_tactile
                    else:
                        extrin = extrin_obs

                    # During supervised training, pass to priv_mlp -> extrin has gt label
                    with torch.no_grad():
                        if 'priv_info' in obs_dict:
                            if self.contact_info:
                                contact_features = self.contact_mlp(obs_dict['contacts'])
                                if self.only_contact:
                                    extrin_gt = contact_features
                                else:
                                    priv_obs = torch.cat([obs_dict['priv_info'], contact_features], dim=-1)
                                    extrin_gt = self.env_mlp(priv_obs)  # extrin
                            else:
                                extrin_gt = self.env_mlp(obs_dict['priv_info'])
                        else:
                            # In case we are evaluating stage2
                            extrin_gt = extrin

                    # extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                    # extrin_gt = torch.tanh(extrin_gt)
                    # extrin = torch.tanh(extrin)

                    # Applying action with student model
                    obs = torch.cat([obs, extrin], dim=-1)

                    # plot for latent viz
                    if display:
                        plt.ylim(-1, 1)
                        plt.scatter(list(range(extrin.shape[-1])), extrin.clone().detach().cpu().numpy()[0, :],
                                    color='r')
                        plt.scatter(list(range(extrin_gt.shape[-1])), extrin_gt.clone().cpu().numpy()[0, :], color='b')
                        plt.pause(0.0001)
                        plt.cla()

                else:
                    # Stage1 -> Getting extrin from the priv_mlp
                    if self.contact_info:
                        contact_features = self.contact_mlp(obs_dict['contacts'])
                        if self.only_contact:
                            extrin = contact_features
                        else:
                            priv_obs = torch.cat([obs_dict['priv_info'], contact_features], dim=-1)
                            extrin = self.env_mlp(priv_obs)
                    else:
                        extrin = self.env_mlp(obs_dict['priv_info'])

                    # extrin = torch.tanh(extrin)

                    # plot for latent viz
                    if display and 'latent' in obs_dict:
                        plt.ylim(-1, 1)
                        plt.scatter(list(range(extrin.shape[-1])), extrin.clone().detach().cpu().numpy()[0, :],
                                    color='b')
                        plt.scatter(list(range(extrin.shape[-1])), obs_dict['latent'].clone().cpu().numpy()[0, :],
                                    color='r')
                        plt.pause(0.0001)
                        plt.cla()

                    obs = torch.cat([obs, extrin], dim=-1)

        x = self.actor_mlp(obs)
        mu = self.mu(x)
        sigma = self.sigma
        if not self.shared_parameters:
            v = self.critic_mlp(obs)
            value = self.value(v)
        else:
            value = self.value(x)

        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result

    # @torch.no_grad()
    def _tactile_encode_multi(self, images):

        #                E, T,(finger) W, H, C  ->   E, T, C, W, H
        left_seq = images[:, :, 0, :, :, :].permute(0, 1, 4, 2, 3)
        right_seq = images[:, :, 1, :, :, :].permute(0, 1, 4, 2, 3)
        bot_seq = images[:, :, 2, :, :, :].permute(0, 1, 4, 2, 3)

        seq = torch.cat((left_seq, right_seq, bot_seq), dim=2)

        emb_multi = self.tactile_decoder_m(seq)

        tactile_embeddings = emb_multi

        tac_emb = self.tactile_mlp(tactile_embeddings)

        return tac_emb

    def _tactile_encode(self, images):

        #                E, T,(finger) W, H, C  ->   E, T, C, W, H
        left_seq = images[:, :, 0, :, :, :].permute(0, 1, 4, 2, 3)
        right_seq = images[:, :, 1, :, :, :].permute(0, 1, 4, 2, 3)
        bot_seq = images[:, :, 2, :, :, :].permute(0, 1, 4, 2, 3)

        emb_left = self.tactile_decoder(left_seq)
        emb_right = self.tactile_decoder(right_seq)
        emb_bottom = self.tactile_decoder(bot_seq)

        tactile_embeddings = torch.cat((emb_left, emb_right, emb_bottom), dim=-1)
        tac_emb = self.tactile_mlp(tactile_embeddings)

        return tac_emb


def load_tactile_resnet(embed_dim, num_channels,
                        root_dir=None, path_checkpoint=None, pre_trained=False):
    import algo.models.convnets.resnets as resnet
    import os

    tactile_encoder = resnet.resnet18(False, False, num_classes=embed_dim,
                                      num_channels=num_channels)

    if pre_trained:
        tactile_encoder.load_state_dict(os.path.join(root_dir, path_checkpoint))
        tactile_encoder.eval()
        for param in tactile_encoder.parameters():
            param.requires_grad = False

    return tactile_encoder

# =======================================================
# ReDi-LPD: Diffusion Student Actor Implementation
# =======================================================
class DiffusionStudentActor(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        self.config = full_config
        # 从 offline_train 配置中读取
        self.train_config = full_config.offline_train
        self.network_config = full_config.train.network
        
        # 1. Action Dimension
        if 'action_dim' in full_config:
            self.action_dim = full_config['action_dim']
        else:
            self.action_dim = 12 
            print("[DiffusionActor] Warning: action_dim not found in config, defaulting to 12")

        # 2. 感知模块
        self.obs_info = self.config.train.ppo["obs_info"]
        self.tactile_info = self.config.train.ppo["tactile_info"]
        self.pcl_info = self.config.train.ppo["pcl_info"]
        self.img_info = self.config.train.ppo["img_info"]
        
        # --- (A) Tactile Encoder ---
        if self.tactile_info:
            tactile_decoder_embed_dim = self.network_config['tactile_decoder_embed_dim']
            tactile_input_dim = self.network_config['tactile_input_dim']
            num_channels = tactile_input_dim[-1]
            # num_fingers = 3 # 移除硬编码
            
            # ResNet Encoder
            # 注意: embed_dim 通常是总维度，除以 3 才是每个 ResNet 的输出
            # 如果是单指输入，我们之后会 duplicate，所以这里依然按 3 指初始化
            self.tactile_decoder = load_tactile_resnet(tactile_decoder_embed_dim // 3, num_channels)
            
            # Tactile MLP Projection
            self.tactile_units = self.network_config["mlp_tactile_units"]
            tactile_input_shape = self.network_config["mlp_tactile_input_shape"]
            self.tactile_mlp = MLP(units=self.tactile_units, input_size=tactile_input_shape)
            
            # Tactile Embedding Dimension
            self.tactile_emb_dim = self.tactile_units[-1]
        else:
            self.tactile_emb_dim = 0

        # --- (B) Proprioception (Student Obs) Encoder ---
        if self.obs_info:
            self.obs_units = self.network_config["obs_units"]
            self.obs_mlp = MLP(units=self.obs_units, input_size=self.network_config["student_obs_input_shape"])
            self.obs_emb_dim = self.obs_units[-1]
        else:
            self.obs_emb_dim = 0

        # --- (C) Point Cloud Encoder ---
        self.pcl_emb_dim = 0
        if self.pcl_info:
            pass 

        # --- (D) Merge Layer (Fusion) ---
        self.cond_dim = self.tactile_emb_dim + self.obs_emb_dim + self.pcl_emb_dim
        
        if self.obs_info and self.tactile_info:
            self.merge_units = self.network_config["merge_units"]
            self.merge_mlp = MLP(
                units=self.merge_units, input_size=self.cond_dim
            )
            self.cond_dim = self.merge_units[-1]

        print(f">>> [ReDi-LPD] Condition Dimension: {self.cond_dim}")

        # 3. 扩散去噪网络
        self.noise_pred_net = DiffusionTransformer(
            action_dim=self.action_dim,
            cond_dim=self.cond_dim,
            embed_dim=256, 
            depth=4        
        )

        # 4. 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2', 
            clip_sample=True,
            prediction_type='epsilon' 
        )

    def get_cond_features(self, obs_dict):
        extrin_tactile = None
        extrin_obs = None
        
        # 1. Encode Tactile
        if self.tactile_info and 'tactile' in obs_dict and obs_dict['tactile'] is not None:
            images = obs_dict['tactile'] 
            extrin_tactile = self._tactile_encode(images)

        # 2. Encode Proprioception (Student Obs)
        if self.obs_info and 'student_obs' in obs_dict and obs_dict['student_obs'] is not None:
            extrin_obs = self.obs_mlp(obs_dict['student_obs'])

        # 3. Fusion
        if self.obs_info and self.tactile_info:
            cond = torch.cat([extrin_tactile, extrin_obs], dim=-1)
            cond = self.merge_mlp(cond)
        elif self.tactile_info:
            cond = extrin_tactile
        elif self.obs_info:
            cond = extrin_obs
        else:
            raise ValueError("No valid observation for diffusion condition!")

        return cond

    def _tactile_encode(self, images):
        # === [ReDi-LPD] 通用自适应触觉编码器 ===
        # 彻底解决 1指/3指/5指不匹配以及维度报错问题
        
        # 1. 维度标准化 -> [B, T, F, H, W, C]
        # ------------------------------------------------
        dims = len(images.shape)
        
        if dims == 6:   # [B, T, F, H, W, C]
            pass
        elif dims == 5: # [B, F, H, W, C] -> 加 Time
            images = images.unsqueeze(1)
        elif dims == 4: # [B, F, H, W] -> 加 Time, Channel
            images = images.unsqueeze(1).unsqueeze(-1)
        else:
            # 这里的报错会直接告诉你维度，非常关键
            raise ValueError(f"[ReDi-LPD] Invalid tactile shape: {images.shape}. Expected 4, 5 or 6 dims.")

        # 2. 自适应手指数量 (Sim优化: 1指 -> 3指)
        # ------------------------------------------------
        # 假设 MLP 期待 3 指 (1536 dim)，但 Sim 只渲染了 1 指
        B, T, F, H, W, C = images.shape
        
        if F == 1:
            # 自动复制 1 指数据填充为 3 指 (或根据配置)
            # 这里硬编码为 3，因为你的 Teacher 模型是 3 指的
            # print(">>> [Auto-Fix] Duplicating 1-finger data to 3 fingers.")
            images = images.repeat(1, 1, 3, 1, 1, 1)
            F = 3 # 更新 F

        # 3. Batch Folding (支持任意 F)
        # ------------------------------------------------
        # Permute: [B, T, F, H, W, C] -> [B, T, F, C, H, W] (ResNet需要 Channel First)
        # 注意：这里假设原始是 (H, W, C)，所以 permute(..., 2, 0, 1) 相对于最后3维
        # 为了兼容之前的 permute(0, 1, 4, 2, 3) 逻辑 -> 对应 C, H, W
        x = images.permute(0, 1, 2, 5, 3, 4) 
        
        # Fold: [B*T*F, C, H, W]
        x_folded = x.reshape(B * T * F, C, H, W)
        
        # 4. ResNet Encoding
        # ------------------------------------------------
        # [B*T*F, Embed]
        features = self.tactile_decoder(x_folded)
        
        # 5. Unfold & Flatten
        # ------------------------------------------------
        # [B, T, F, Embed]
        features = features.reshape(B, T, F, -1)
        
        # [B, T, F*Embed] -> 这一步自动处理了拼接，无论 F 是几
        tactile_embeddings = features.reshape(B, T, -1)
        
        # ReDi-LPD: 取最后一帧
        current_embeddings = tactile_embeddings[:, -1, :]
        
        # 6. Final MLP
        # ------------------------------------------------
        tac_emb = self.tactile_mlp(current_embeddings)
        
        return tac_emb

    def compute_loss(self, obs_dict, teacher_action):
        cond = self.get_cond_features(obs_dict)
        device = cond.device
        batch_size = cond.shape[0]
        noise = torch.randn(batch_size, self.action_dim, device=device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(teacher_action, noise, timesteps)
        pred_noise = self.noise_pred_net(noisy_action, timesteps, cond)
        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def predict(self, obs_dict, requires_grad=False):
        action = self.act(obs_dict)
        return action, None 

    @torch.no_grad()
    def act(self, obs_dict):
        device = self.device if hasattr(self, 'device') else list(self.parameters())[0].device
        cond = self.get_cond_features(obs_dict)
        batch_size = cond.shape[0]
        action = torch.randn(batch_size, self.action_dim, device=device)
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.noise_pred_net(action, timesteps, cond)
            action = self.noise_scheduler.step(pred_noise, t, action).prev_sample
        return action