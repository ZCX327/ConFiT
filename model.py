import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import importlib
from Temporal.e2fgvi_hq import InpaintGenerator
from Spatial.mat import Generator
from util.ConfidenceMaskGeneratorSPy import ConfidenceMaskGeneratorSPy
from util.FusionNet import MaskGuidedTemporalCombinerDyn
def decompose_lh(image, kernel_size=15):
    """分离低频和高频"""
    # 低频 = 平滑版本（高斯模糊）
    blurred = F.avg_pool2d(image, kernel_size, padding=kernel_size//2, stride=1)
    # 高频 = 原图 - 低频
    high_freq = image - blurred
    return blurred, high_freq

def gaussian_blur_mask(mask, kernel_size=10, sigma=10):
    """
    对 mask 做高斯平滑
    输入支持:
      - (B, 1, H, W)
      - (B, T, 1, H, W)
    输出形状与输入一致
    """
    import kornia

    if mask.dim() == 5:  # (B, T, 1, H, W)
        B, T, C, H, W = mask.shape
        mask = mask.view(B * T, C, H, W)  # 展平 T
        out = kornia.filters.gaussian_blur2d(mask, (kernel_size, kernel_size), (sigma, sigma))
        out = out.view(B, T, C, H, W)     # 还原形状
    elif mask.dim() == 4:  # (B, 1, H, W)
        out = kornia.filters.gaussian_blur2d(mask, (kernel_size, kernel_size), (sigma, sigma))
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}, expected 4D or 5D tensor.")

    return out


class ConfidenceFusion3D(nn.Module):
    def __init__(self, t_window=5, alpha_high=0.1, alpha_low=0.1):
        super().__init__()
        self.t_window = t_window
        self.alpha_high = nn.Parameter(torch.tensor(alpha_high, dtype=torch.float32))
        self.alpha_low  = nn.Parameter(torch.tensor(alpha_low, dtype=torch.float32))

        self.temporal_conv = nn.Sequential(
            nn.Conv3d(9, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.refine_2d = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1)
        )

        self.final_refine = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 3, padding=1)
        )


    def forward(self, out_mat_seq, out_e2fgvi_seq, mask_seq=None, M_high_seq=None, M_low_seq=None):
        
        B, T, C, H, W = out_mat_seq.shape
        device = out_mat_seq.device
        eps = 1e-6

        if M_high_seq is None: M_high_seq = torch.zeros(B, T, 1, H, W, device=device)
        if M_low_seq  is None: M_low_seq  = torch.zeros(B, T, 1, H, W, device=device)
        if mask_seq   is None: mask_seq   = torch.clamp(M_high_seq + M_low_seq, 0, 1)

        fused_seq = []
        for t in range(T):
            # --- 构造时序窗口 ---
            t_start, t_end = 0, T
            mat_win, e2f_win = out_mat_seq[:, t_start:t_end], out_e2fgvi_seq[:, t_start:t_end]
            high_win, low_win, mask_win = M_high_seq[:, t_start:t_end], M_low_seq[:, t_start:t_end], mask_seq[:, t_start:t_end]

            # --- 高低频引导 + 堆叠输入 ---
            fused_chs = []
            for i in range(t_end - t_start):
                low_e2f, high_e2f = decompose_lh(e2f_win[:, i])
                low_mat, high_mat = decompose_lh(mat_win[:, i])
                Mh, Ml, Mk = high_win[:, i]*mask_win[:, i], low_win[:, i]*mask_win[:, i], mask_win[:, i]
                fused = Mh*(low_e2f+high_e2f) + Ml*(low_e2f+high_mat)
                fused_chs.append(torch.cat([fused, mat_win[:, i], e2f_win[:, i]], dim=1))

            fused_win = torch.stack(fused_chs, dim=2)            # (B,9,Tw,H,W)
            feat_3d   = self.temporal_conv(fused_win)            # (B,32,Tw,H,W)
            refined   = self.refine_2d(feat_3d[:, :, feat_3d.size(2)//2])  # 中心帧输出

            # --- 权重融合 ---
            mat_c, e2f_c = out_mat_seq[:, t], out_e2fgvi_seq[:, t]
            Mh, Ml = M_high_seq[:, t], M_low_seq[:, t]
            
            # 低置信区域使用 refined 作为 MAT 引导
            w_e2  = self.alpha_high * Mh          # 高置信 -> E2FGVI
            w_mat = self.alpha_low  * Ml          # 低置信 -> refined
            w_mid = 1.0 - Mh - Ml                 # 剩余区域 -> 当前 MAT
            
            # 避免除0
            denom = w_e2 + w_mat + w_mid + 1e-6
            w_e2 /= denom
            w_mat /= denom
            w_mid /= denom
            
            # 最终融合
            guided = (w_e2.expand_as(e2f_c)  * e2f_c +
                      w_mat.expand_as(mat_c) * mat_c +
                      w_mid.expand_as(mat_c)   * mat_c)

            
            fused_seq.append(guided)

        fused_seq = torch.stack(fused_seq, dim=1)
        
        fused_seq  = fused_seq * M_low_seq + out_e2fgvi_seq * (1.0 - M_low_seq)
        
         # ---------- 最终 refine ----------
        fused_seq = fused_seq.view(B*T, C, H, W)
        fused_seq = 0.1 * self.final_refine(fused_seq) + fused_seq
        fused_seq = fused_seq.view(B, T, C, H, W)

        return fused_seq  # (B,T,3,H,W)




def load_mat():
    """
    从 MAT 的 Generator 里只提取 SynthesisNet (相当于 encoder+decoder)。
    """
    # pkl_path = '../release_model/gen_mat_only.pth'
    mat_model = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3)
    # ckpt = torch.load(pkl_path)
    # mat_model.load_state_dict(ckpt, strict=False) 
    # print(f"mat from : {pkl_path}")
    return mat_model


def load_e2fgvi():
    pkl_path2 = '../model_pkl/e2fgvi_ckl.pth'
    model = InpaintGenerator()
    data = torch.load(pkl_path2)
    model.load_state_dict(data)
    print(f"e2fgvi from : {pkl_path2}")
    return model


# =======================
# 融合网络框架
# =======================
class ConFit(nn.Module):
    def __init__(self, mod = None):
        super().__init__()
        self.mod = mod
        self.mat_model = load_mat()      # MAT 单帧补全器
        self.e2fgvi_model = load_e2fgvi()  # E2FGVI 视频补全器
        # self.fusion = ConfidenceFusion3D()
        self.fusionT = MaskGuidedTemporalCombinerDyn()
        self.conf_gen = ConfidenceMaskGeneratorSPy()
        if mod == 1:
            for param in self.e2fgvi_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.e2fgvi_model.eval()
        elif mod == 2:
            for param in self.mat_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.mat_model.eval()
        elif mod == 3:
            for param in self.e2fgvi_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.e2fgvi_model.eval()
            for param in self.mat_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.mat_model.eval()


        

    def forward(self, masked_frames, masks, z):
        """
        masked_frames: (B, T, 3, H, W) 输入视频片段
        masks: (B, T, 1, H, W) 遮挡掩膜
        conf_mask: (B, T, 1, H, W) 高置信引导
        """
        B, T, C, H, W = masked_frames.size()

        M_high_all, M_low_all, _ = self.conf_gen(masked_frames,masks)
        M_high_all = gaussian_blur_mask(M_high_all, kernel_size=7, sigma=7)
        M_low_all  = gaussian_blur_mask(M_low_all,  kernel_size=7, sigma=7)
        mask_smooth = gaussian_blur_mask(masks,      kernel_size=7, sigma=7)
        # === MAT 单帧补全
        mat_outs = []


        if self.mod == 2 or self.mod == 3:
            with torch.no_grad():
                for t in range(T):
                    frame = masked_frames[:, t]   # (B, 3, H, W)
                    mask = masks[:, t]
                    batch_z = z[:,t]
                    mat_out = self.mat_model(frame, mask, batch_z, None)
                    mat_outs.append(mat_out)
                mat_outs = torch.stack(mat_outs, dim=1)  # (B, T, 3, H, W)
        else:
            for t in range(T):
                frame = masked_frames[:, t]   # (B, 3, H, W)
                mask = masks[:, t]
                batch_z = z[:,t]
                mat_out = self.mat_model(frame, mask, batch_z, None)
                mat_outs.append(mat_out)
            mat_outs = torch.stack(mat_outs, dim=1)  # (B, T, 3, H, W)



        mat_first = mat_outs[:, 0:1]  # (B,1,3,H,W)
        masked_frames = torch.cat([masked_frames, mat_first], dim=1)  # (B,T+1,3,H,W)

        if self.mod == 1 or self.mod == 3:
            with torch.no_grad():
                e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T + 1, 3, H, W)
            e2fgvi_outs = e2fgvi_outs[:, :T]  # (B, T, 3, H, W)
            
        else:
            e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T + 1, 3, H, W)
            e2fgvi_outs = e2fgvi_outs[:, :T]  # (B, T, 3, H, W)
         
        # === 融合阶段（改成 3D 融合）===

        fused_outs,_ = self.fusionT(mat_outs, e2fgvi_outs, M_high_all, M_low_all)  # (B,T,3,H,W)


        return fused_outs, mat_outs, e2fgvi_outs, M_high_all ,M_low_all



# =======================
# 融合网络框架
# =======================
class E2FGVI(nn.Module):
    def __init__(self, mod = None):
        super().__init__()
        self.mod = mod
        self.e2fgvi_model = load_e2fgvi()  # E2FGVI 视频补全器
        self.conf_gen = ConfidenceMaskGeneratorSPy()


    def forward(self, masked_frames, masks, z):
        """
        masked_frames: (B, T, 3, H, W) 输入视频片段
        masks: (B, T, 1, H, W) 遮挡掩膜
        conf_mask: (B, T, 1, H, W) 高置信引导
        """
        B, T, C, H, W = masked_frames.size()

        M_high_all, M_low_all, _ = self.conf_gen(masked_frames,masks)

        # === E2FGVI 视频补全
        if self.mod == 1 or self.mod == 3:
            with torch.no_grad():
                e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T, 3, H, W)
        else:
            e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T, 3, H, W)
            
        masked_frames = (M_low_all) * e2fgvi_outs
        
        # === E2FGVI 视频补全
        if self.mod == 1 or self.mod == 3:
            with torch.no_grad():
                e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T, 3, H, W)
        else:
            e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T, 3, H, W)
         
         

        return e2fgvi_outs



# =======================
# 融合网络框架
# =======================
class MAT(nn.Module):
    def __init__(self, mod = None):
        super().__init__()
        self.mod = mod
        self.mat_model = load_mat()      # MAT 单帧补全器
        

    def forward(self, masked_frames, masks, z):
        """
        masked_frames: (B, T, 3, H, W) 输入视频片段
        masks: (B, T, 1, H, W) 遮挡掩膜
        conf_mask: (B, T, 1, H, W) 高置信引导
        """
        B, T, C, H, W = masked_frames.size()
        # === MAT 单帧补全
        mat_outs = []
        if self.mod == 2 or self.mod == 3:
            with torch.no_grad():
                for t in range(T):
                    frame = masked_frames[:, t]   # (B, 3, H, W)
                    mask = masks[:, t]
                    batch_z = z[:,t]
                    batch_img_resized = F.interpolate(frame, size=(512, 512), mode='nearest')
                    batch_mask_resized = F.interpolate(mask, size=(512, 512), mode='nearest')
                    mat_out = self.mat_model(batch_img_resized, batch_mask_resized, batch_z, None)
                    mat_out_up = F.interpolate(mat_out, size=(540, 540), mode='nearest')
                    mat_outs.append(mat_out_up)
                mat_outs = torch.stack(mat_outs, dim=1)  # (B, T, 3, H, W)
        else:
            for t in range(T):
                frame = masked_frames[:, t]   # (B, 3, H, W)
                mask = masks[:, t]
                batch_z = z[:,t]
                batch_img_resized = F.interpolate(frame, size=(512, 512), mode='nearest')
                batch_mask_resized = F.interpolate(mask, size=(512, 512), mode='nearest')
                mat_out = self.mat_model(batch_img_resized, 1.0 - batch_mask_resized, batch_z, None)
                mat_out_up = F.interpolate(mat_out, size=(540, 540), mode='nearest')
                mat_outs.append(mat_out_up)
            mat_outs = torch.stack(mat_outs, dim=1)  # (B, T, 3, H, W)


        return mat_outs



import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.layers = nn.Sequential(
            # Layer 1: k=3, s=2, p=1 → H/2
            nn.Conv2d(in_channels, ndf, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: k=3, s=2, p=1 → H/4
            nn.Conv2d(ndf, ndf * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: k=3, s=2, p=1 → H/8
            nn.Conv2d(ndf * 2, ndf * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: k=3, s=1, p=1 → H/8 (保持尺寸)
            nn.Conv2d(ndf * 4, ndf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 输出层，k=3, s=1, p=1
            nn.Conv2d(ndf * 8, 1, 3, stride=1, padding=1)
            # 输出: (B, 1, H//8, W//8)
        )

    def forward(self, x):
        return self.layers(x)


# =======================
# 融合网络框架
# =======================
class MAT_E2FGVI_FusionNetNEW(nn.Module):
    def __init__(self, mod = None):
        super().__init__()
        self.mod = mod
        self.mat_model = load_mat()      # MAT 单帧补全器
        self.e2fgvi_model = load_e2fgvi()  # E2FGVI 视频补全器
        self.fusion = ConfidenceFusion3D()
        self.conf_gen = ConfidenceMaskGeneratorSPy()
        if mod == 1:
            for param in self.e2fgvi_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.e2fgvi_model.eval()
        elif mod == 2:
            for param in self.mat_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.mat_model.eval()
        elif mod == 3:
            for param in self.e2fgvi_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.e2fgvi_model.eval()
            for param in self.mat_model.parameters():
                param.requires_grad = False 
            # # 可选：设置为 eval 模式
            self.mat_model.eval()


        

    def forward(self, masked_frames, masks, z):
        """
        masked_frames: (B, T, 3, H, W) 输入视频片段
        masks: (B, T, 1, H, W) 遮挡掩膜
        conf_mask: (B, T, 1, H, W) 高置信引导
        """
        B, T, C, H, W = masked_frames.size()

        M_high_all, M_low_all, _ = self.conf_gen(masked_frames,masks)
        M_high_all = gaussian_blur_mask(M_high_all, kernel_size=7, sigma=7)
        M_low_all  = gaussian_blur_mask(M_low_all,  kernel_size=7, sigma=7)
        mask_smooth = gaussian_blur_mask(masks,      kernel_size=7, sigma=7)



        if self.mod == 1 or self.mod == 3:
            with torch.no_grad():
                e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs[:, :T]  # (B, T, 3, H, W)
            
        else:
            e2fgvi_outs, e2f_flow = self.e2fgvi_model(masked_frames, num_local_frames=T)  # (B*T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs.view(B, T, 3, H, W)
            e2fgvi_outs = e2fgvi_outs[:, :T]  # (B, T, 3, H, W)

        # === MAT 单帧补全
        mat_outs = []

        if self.mod == 2 or self.mod == 3:
            with torch.no_grad():
                for t in range(T):
                    frame = e2fgvi_outs[:, t]   # (B, 3, H, W)
                    mask = masks[:, t]
                    batch_z = z[:,t]
                    mat_out = self.mat_model(frame, mask, batch_z, None)
                    mat_outs.append(mat_out)
                mat_outs = torch.stack(mat_outs, dim=1)  # (B, T, 3, H, W)
        else:
            for t in range(T):
                frame = e2fgvi_outs[:, t]   # (B, 3, H, W)
                mask = masks[:, t]
                batch_z = z[:,t]
                mat_out = self.mat_model(frame, mask, batch_z, None)
                mat_outs.append(mat_out)
            mat_outs = torch.stack(mat_outs, dim=1)  # (B, T, 3, H, W)


  
         
        # === 融合阶段（改成 3D 融合）===
        fused_outs = self.fusion(mat_outs, e2fgvi_outs, mask_smooth, M_high_all, M_low_all)  # (B,T,3,H,W)

        return fused_outs, mat_outs, e2fgvi_outs, M_high_all ,M_low_all