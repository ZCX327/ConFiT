import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceFusion3D2(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        # 统一编码器（2D）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),  # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),  # H/8
            nn.ReLU(inplace=True)
        )

        # 3D 卷积模块（只在融合后使用）
        self.conv3d = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
        )

        # 解码器（2D）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),  # H/4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, 3, 4, stride=2, padding=1),  # H
            nn.Tanh()
        )

    def forward(self, mat_outs, e2fgvi_outs, masks, M_high_all, M_low_all):
        """
        mat_outs:    (B,T,3,H,W)
        e2fgvi_outs: (B,T,3,H,W)
        masks:       (B,T,1,H,W)
        M_high_all:  (B,T,1,H,W)
        M_low_all:   (B,T,1,H,W)
        """
        B, T, C, H, W = mat_outs.shape
        fused_feats = []

        for t in range(T):
            mat = mat_outs[:, t]      # (B,3,H,W)
            e2f = e2fgvi_outs[:, t]  # (B,3,H,W)

            feat_mat = self.encoder(mat)
            feat_e2f = self.encoder(e2f)
            B, C, Hf, Wf = feat_mat.shape
            M = F.interpolate(masks[:, t], size=(Hf, Wf), mode="nearest")
            Mh = F.interpolate(M_high_all[:, t], size=(Hf, Wf), mode="nearest")
            Ml = F.interpolate(M_low_all[:, t], size=(Hf, Wf), mode="nearest")


            fused_feat = feat_e2f * M + Mh * feat_e2f + Ml * feat_mat  # (B,C,H/8,W/8)
            fused_feats.append(fused_feat)

        # 拼接成 3D 特征 (B,C,T,H/8,W/8)
        fused_feats = torch.stack(fused_feats, dim=2)

        # 在遮挡区域应用 3D 卷积建模时序
        fused_feats = self.conv3d(fused_feats)

        # 再逐帧解码
        fused_seq = []
        for t in range(T):
            out = self.decoder(fused_feats[:, :, t])
            fused_seq.append(out)

        fused_seq = torch.stack(fused_seq, dim=1)  # (B,T,3,H,W)
        return fused_seq


def tensor_stats(tensor, name="tensor"):
    """
    输出 tensor 的统计信息
    tensor: torch.Tensor
    name: 名称，用于打印
    """
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    t_mean = tensor.mean().item()
    t_median = tensor.median().item()
    t_std = tensor.std().item()
    print(f"[{name}] shape: {tensor.shape}")
    print(f"  min: {t_min:.4f}, max: {t_max:.4f}, mean: {t_mean:.4f}, median: {t_median:.4f}, std: {t_std:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGuidedTemporalCombiner(nn.Module):
    """
    学习如何使用高/低置信掩膜融合两路已有输出（F_t: E2FGVI, F_s: MAT）。
    输入：
        F_t, F_s: (B, T, C, H, W)
        M_high, M_low: (B, T, 1, H, W) 或 (B, T, C, H, W) (值可为 soft)
    输出：
        fused: (B, T, C, H, W)
        alpha: (B, T, 1, H, W)  # alpha 表示对 F_t 的权重 (0..1)
    Notes:
        - alpha 越接近 1 表示更信任 F_t（E2FGVI）；越接近 0 表示更信任 F_s（MAT）。
        - 若显存受限，可降低 hidden_ch 或采用分块(window)时序 attention。
    """
    def __init__(self, in_ch=3, hidden_ch=64, heads=4, attn_dropout=0.0, normalize_masks=True):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.heads = heads
        self.normalize_masks = normalize_masks

        # 输入通道：F_t( C ) + F_s( C ) + M_high(1) + M_low(1)
        fusion_in = in_ch * 2 + 2

        # 空间编码器（保留 H,W）
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(fusion_in, hidden_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # 时序注意力：在每个像素位置上做 T 长度的 self-attention
        # 使用 nn.MultiheadAttention (batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_ch, num_heads=heads,
                                                   dropout=attn_dropout, batch_first=True)

        # 将 attn 上的 hidden 映射到 alpha logits (1通道)
        self.to_alpha = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch//2 if hidden_ch>=2 else 1, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch//2 if hidden_ch>=2 else 1, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def _expand_mask(self, M, C, device, dtype):
        # 扩展 mask 到 (B,T,C,H,W) 方便广播
        if M is None:
            return torch.zeros(1, device=device, dtype=dtype)  # not used
        if M.dim() == 5 and M.size(2) == 1:
            return M.expand(-1, -1, C, -1, -1)
        return M  # 已经是 (B,T,C,H,W)

    def forward(self, F_t, F_s, M_high, M_low):
        """
        返回: fused, alpha  (alpha 对应 F_t 的权重)
        """
        assert F_t.shape == F_s.shape, "F_t 和 F_s 形状必须一致"
        B, T, C, H, W = F_t.shape
        device = F_t.device
        dtype = F_t.dtype

        # ----- 规范化 masks -----
        # 将 M_high/M_low 扩展到通道维 (B,T,C,H,W)
        M_high_c = self._expand_mask(M_high, C, device, dtype)  # (B,T,C,H,W)
        M_low_c  = self._expand_mask(M_low,  C, device, dtype)

        # clamp 到 [0,1]
        M_high_c = torch.clamp(M_high_c.float(), 0.0, 1.0)
        M_low_c  = torch.clamp(M_low_c.float(),  0.0, 1.0)

        if self.normalize_masks:
            total = (M_high_c + M_low_c)
            # 若 total>1，则按比例缩放到 1（逐像素）
            scale = torch.where(total > 1.0, 1.0 / (total + 1e-6), torch.ones_like(total))
            M_high_c = M_high_c * scale
            M_low_c  = M_low_c  * scale

        # ----- 构造编码器输入（并行做 B*T） -----
        # 合并通道：F_t, F_s, M_high(1), M_low(1)
        Mh_1 = M_high if (M_high is not None and M_high.dim()==5 and M_high.size(2)==1) else M_high_c[:, :, :1]
        Ml_1 = M_low  if (M_low  is not None and M_low.dim()==5 and M_low.size(2)==1) else M_low_c[:, :, :1]
        enc_in = torch.cat([F_t, F_s, Mh_1, Ml_1], dim=2)  # (B, T, 2C+2, H, W)
        enc_flat = enc_in.view(B*T, -1, H, W)               # (B*T, 2C+2, H, W)

        # ----- 空间编码 -----
        feat = self.spatial_encoder(enc_flat)  # (B*T, hidden, H, W)

        # ----- reshape 为 (B, H*W, T, hidden) 并做时序 attention -----
        # (B*T, hidden, H, W) -> (B, T, hidden, H, W)
        feat_bt = feat.view(B, T, self.hidden_ch, H, W)
        # 转为 (B, H*W, T, hidden)
        feat_pos = feat_bt.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, hidden)
        HW = H * W
        feat_pos = feat_pos.view(B, HW, T, self.hidden_ch)      # (B, HW, T, hidden)

        # 为 MultiheadAttention 调整形状 (合并 B*HW 作为 batch)
        feat_attn_input = feat_pos.view(B*HW, T, self.hidden_ch)  # (B*HW, T, hidden)
        # Apply temporal self-attention (per spatial position)
        attn_out, _ = self.temporal_attn(feat_attn_input, feat_attn_input, feat_attn_input)
        attn_out = attn_out.view(B, HW, T, self.hidden_ch)  # (B, HW, T, hidden)

        # 恢复回 (B, T, hidden, H, W)
        attn_out = attn_out.view(B, H, W, T, self.hidden_ch).permute(0, 3, 4, 1, 2).contiguous()
        # -> (B, T, hidden, H, W)
        attn_out = attn_out.view(B*T, self.hidden_ch, H, W)

        # ----- 由 attn_out 计算 alpha（0..1） -----
        alpha_map = self.to_alpha(attn_out)  # (B*T, 1, H, W)
        alpha_map = alpha_map.view(B, T, 1, H, W)  # (B,T,1,H,W)

        # ----- final fusion: out = alpha * F_t + (1-alpha) * F_s -----
        alpha_c = alpha_map.expand(-1, -1, C, -1, -1)  # broadcast to channels
        fused = alpha_c * F_t + (1.0 - alpha_c) * F_s
        tensor_stats(alpha_c,"alpha_c")

        # print(f"alpha_c:{alpha_c.shape}")


        return fused, alpha_map


class MaskGuidedTemporalCombinerDyn(nn.Module):
    """
    学习如何使用高/低置信掩膜融合两路已有输出（F_t: E2FGVI, F_s: MAT）。
    输入：
        F_t, F_s: (B, T, C, H, W)
        M_high, M_low: (B, T, 1, H, W) 或 (B, T, C, H, W) (值可为 soft)
    输出：
        fused: (B, T, C, H, W)
        alpha: (B, T, 1, H, W)  # alpha 表示对 F_t 的权重 (0..1)
    Notes:
        - alpha 越接近 1 表示更信任 F_t（E2FGVI）；越接近 0 表示更信任 F_s（MAT）。
        - 若显存受限，可降低 hidden_ch 或采用分块(window)时序 attention。
    """
    def __init__(self, in_ch=3, hidden_ch=64, heads=4, attn_dropout=0.0, normalize_masks=True):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.heads = heads
        self.normalize_masks = normalize_masks

        # 输入通道：F_t( C ) + F_s( C ) + M_high(1) + M_low(1)
        fusion_in = in_ch * 2 + 2

        # 空间编码器（保留 H,W）
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(fusion_in, hidden_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # 时序注意力：在每个像素位置上做 T 长度的 self-attention
        # 使用 nn.MultiheadAttention (batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_ch, num_heads=heads,
                                                   dropout=attn_dropout, batch_first=True)

        # 将 attn 上的 hidden 映射到 alpha logits (1通道)
        self.to_alpha = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch//2 if hidden_ch>=2 else 1, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch//2 if hidden_ch>=2 else 1, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def _expand_mask(self, M, C, device, dtype):
        # 扩展 mask 到 (B,T,C,H,W) 方便广播
        if M is None:
            return torch.zeros(1, device=device, dtype=dtype)  # not used
        if M.dim() == 5 and M.size(2) == 1:
            return M.expand(-1, -1, C, -1, -1)
        return M  # 已经是 (B,T,C,H,W)

    def forward(self, F_t, F_s, M_high, M_low):
        """
        返回: fused, alpha  (alpha 对应 F_t 的权重)
        """
        assert F_t.shape == F_s.shape, "F_t 和 F_s 形状必须一致"
        B, T, C, H, W = F_t.shape
        device = F_t.device
        dtype = F_t.dtype

        # ----- 规范化 masks -----
        # 将 M_high/M_low 扩展到通道维 (B,T,C,H,W)
        M_high_c = self._expand_mask(M_high, C, device, dtype)  # (B,T,C,H,W)
        M_low_c  = self._expand_mask(M_low,  C, device, dtype)

        # clamp 到 [0,1]
        M_high_c = torch.clamp(M_high_c.float(), 0.0, 1.0)
        M_low_c  = torch.clamp(M_low_c.float(),  0.0, 1.0)

        if self.normalize_masks:
            total = (M_high_c + M_low_c)
            # 若 total>1，则按比例缩放到 1（逐像素）
            scale = torch.where(total > 1.0, 1.0 / (total + 1e-6), torch.ones_like(total))
            M_high_c = M_high_c * scale
            M_low_c  = M_low_c  * scale

        # ----- 构造编码器输入（并行做 B*T） -----
        # 合并通道：F_t, F_s, M_high(1), M_low(1)
        Mh_1 = M_high if (M_high is not None and M_high.dim()==5 and M_high.size(2)==1) else M_high_c[:, :, :1]
        Ml_1 = M_low  if (M_low  is not None and M_low.dim()==5 and M_low.size(2)==1) else M_low_c[:, :, :1]
        enc_in = torch.cat([F_t, F_s, Mh_1, Ml_1], dim=2)  # (B, T, 2C+2, H, W)
        enc_flat = enc_in.view(B*T, -1, H, W)               # (B*T, 2C+2, H, W)

        # ----- 空间编码 -----
        feat = self.spatial_encoder(enc_flat)  # (B*T, hidden, H, W)

        # ----- reshape 为 (B, H*W, T, hidden) 并做时序 attention -----
        # (B*T, hidden, H, W) -> (B, T, hidden, H, W)
        feat_bt = feat.view(B, T, self.hidden_ch, H, W)
        # 转为 (B, H*W, T, hidden)
        feat_pos = feat_bt.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, hidden)
        HW = H * W
        feat_pos = feat_pos.view(B, HW, T, self.hidden_ch)      # (B, HW, T, hidden)

        # 为 MultiheadAttention 调整形状 (合并 B*HW 作为 batch)
        feat_attn_input = feat_pos.view(B*HW, T, self.hidden_ch)  # (B*HW, T, hidden)
        # Apply temporal self-attention (per spatial position)
        attn_out, _ = self.temporal_attn(feat_attn_input, feat_attn_input, feat_attn_input)
        attn_out = attn_out.view(B, HW, T, self.hidden_ch)  # (B, HW, T, hidden)

        # 恢复回 (B, T, hidden, H, W)
        attn_out = attn_out.view(B, H, W, T, self.hidden_ch).permute(0, 3, 4, 1, 2).contiguous()
        # -> (B, T, hidden, H, W)
        attn_out = attn_out.view(B*T, self.hidden_ch, H, W)
        

        alpha_init = 0.7 * M_high_c[:, :, :1] + 0.3 * (1.0 - M_low_c[:, :, :1])  # (B,T,1,H,W)
        # ----- 由 attn_out 计算 alpha（0..1） -----
        alpha_map = self.to_alpha(attn_out)  # (B*T, 1, H, W)
        alpha_map = alpha_map.view(B, T, 1, H, W)  # (B,T,1,H,W)
        
        guided_weight = 0.3  # 可调，0.5 表示 alpha_map 和 alpha_init 各占一半
        alpha_map = alpha_map * (1 - guided_weight) + alpha_init * guided_weight
        alpha_map = torch.clamp(alpha_map, 0.0, 1.0)

        
        # ----- final fusion: out = alpha * F_t + (1-alpha) * F_s -----
        alpha_c = alpha_map.expand(-1, -1, C, -1, -1)  # broadcast to channels
        fused = alpha_c * F_t + (1.0 - alpha_c) * F_s

        return fused, alpha_map

