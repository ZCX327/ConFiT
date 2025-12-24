import torch
import torch.nn as nn
from model.modules.flow_comp import SPyNet, flow_warp

class ConfidenceMaskGeneratorSPy(nn.Module):
    """
    使用 SPyNet 估计光流，通过 warp 可见性掩膜（mask），
    判断遮挡区域中哪些部分可以从邻近帧可靠恢复（高置信），哪些不能（低置信）。
    支持前后多帧参考。
    额外返回 mask 交集，用于表示完全无法恢复区域（对所有帧一致）。
    """
    def __init__(self, neighbor_range=2, thresh=0.2, fuse_mode="mean"):
        """
        Args:
            neighbor_range: int, 使用多少邻近帧
            thresh: float, 可恢复性阈值
            fuse_mode: str, 融合方式 ["mean", "max", "weighted"]
        """
        super().__init__()
        self.flow_model = SPyNet(use_pretrain=True).eval()
        for p in self.flow_model.parameters():
            p.requires_grad = False
        self.neighbor_range = neighbor_range
        self.thresh = thresh
        self.fuse_mode = fuse_mode

    def forward(self, frames, masks, thresh=None):
        """
        Args:
            frames: [B, T, C, H, W], range [0,1]
            masks:  [B, T, 1, H, W], 1=可见, 0=遮挡
            thresh: float, 可恢复性阈值

        Returns:
            M_high: [B, T, 1, H, W], 高置信掩膜
            M_low:  [B, T, 1, H, W], 低置信掩膜
            M_inter_all: [B, T, 1, H, W], 全局交集
        """
        if thresh is None:
            thresh = self.thresh

        b, t, c, h, w = frames.shape
        M_high_all, M_low_all = [], []

        # ===== 全局交集（所有帧遮挡区域的交集） =====
        inter_conf_list = [1 - masks[:, i] for i in range(t)]
        M_intersection = torch.stack(inter_conf_list, dim=0).min(0)[0]  # [B,1,H,W]
        M_inter_all = M_intersection.unsqueeze(1).repeat(1, t, 1, 1, 1)

        for i in range(t):
            current_frame = frames[:, i]
            current_mask = masks[:, i]
            occlusion_region = 1 - current_mask  # 当前帧被遮挡部分

            # ===== 高置信区域：warp邻居mask融合 =====
            warped_list, weights = [], []
            for offset in range(-self.neighbor_range, self.neighbor_range + 1):
                if offset == 0:
                    continue
                j = i + offset
                if j < 0 or j >= t:
                    continue

                neighbor_frame = frames[:, j]
                neighbor_mask = masks[:, j]

                flow_neighbor_to_curr = self.flow_model(neighbor_frame, current_frame)
                warped_neighbor_mask = flow_warp(
                    neighbor_mask, flow_neighbor_to_curr.permute(0, 2, 3, 1)
                )
                warped_neighbor_mask = (warped_neighbor_mask > 0.5).float()

                if self.fuse_mode == "weighted":
                    weight = 1.0 / abs(offset)
                else:
                    weight = 1.0
                warped_list.append(warped_neighbor_mask * weight)
                weights.append(weight)

            if warped_list:
                stacked = torch.stack(warped_list, dim=0)  # [N,B,1,H,W]
                if self.fuse_mode == "mean":
                    fused_warped_mask = stacked.mean(dim=0)
                elif self.fuse_mode == "max":
                    fused_warped_mask = stacked.max(dim=0)[0]
                elif self.fuse_mode == "weighted":
                    fused_warped_mask = stacked.sum(dim=0) / sum(weights)
                else:
                    raise ValueError(f"Unknown fuse_mode: {self.fuse_mode}")
            else:
                fused_warped_mask = torch.zeros_like(current_mask)

            # 高置信 = 遮挡区域 ∩ (warp 可恢复区域)
            M_high = (fused_warped_mask > thresh).float() * occlusion_region
            # 低置信 = 遮挡区域 - 高置信（严格互补）
            M_low = occlusion_region - M_high
            M_low = torch.clamp(M_low, 0, 1)

            M_high_all.append(M_high)
            M_low_all.append(M_low)

        M_high_all = torch.stack(M_high_all, dim=1)
        M_low_all = torch.stack(M_low_all, dim=1)

        return M_high_all, M_low_all, M_inter_all
