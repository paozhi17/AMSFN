"""AMSFN modules: ARFAM (A), CSSAM (B), DFRM (C) and their insertion into YOLO11.

YOLO11 layer indices (identical for n/s/m/l/x):
  A  ARFAM  : after backbone C3k2 layers 4, 6, 8
  B  CSSAM  : replaces the neck Concat layers 12 and 15
  C  DFRM   : DySample replaces nn.Upsample (layers 11, 14);
              EMA + SE on the neck outputs (layers 16, 19, 22)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


# ---------------------------------------------------------------- A. ARFAM
class ARFAM(nn.Module):
    """Adaptive Receptive Field Alignment Module (Eq. 1-5).

    Experts: 7x7, 3x3 and 3x3 (dilation 2) depth-wise kernels, embedded in a
    common 7x7 support and mixed per sample by softmax routing.
    """

    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        self.c = c2
        self.compress = Conv(c1, c2, 1)
        self.router_dw = nn.Conv2d(c2, c2, 7, padding=3, groups=c2)
        self.router_fc = nn.Linear(c2, 3)
        self.w7 = nn.Parameter(torch.empty(c2, 1, 7, 7))
        self.w3 = nn.Parameter(torch.empty(c2, 1, 3, 3))
        self.w3d = nn.Parameter(torch.empty(c2, 1, 3, 3))
        for w in (self.w7, self.w3, self.w3d):
            nn.init.kaiming_normal_(w, mode="fan_out")
        self.bias = nn.Parameter(torch.zeros(c2))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

    def _kernel(self, pi):
        e1 = self.w7
        e2 = F.pad(self.w3, (2, 2, 2, 2))
        e3 = self.w3.new_zeros(self.c, 1, 7, 7)
        e3[:, :, 1:6:2, 1:6:2] = self.w3d
        experts = torch.stack([e1, e2, e3], 0)
        return torch.einsum("bk,kcijh->bcijh", pi.to(experts.dtype), experts)

    def forward(self, x):
        B, _, H, W = x.shape
        xp = self.compress(x)
        pi = self.router_fc(self.router_dw(xp).mean((2, 3))).softmax(-1)
        k = self._kernel(pi).reshape(B * self.c, 1, 7, 7)
        y = F.conv2d(xp.reshape(1, B * self.c, H, W), k, padding=3, groups=B * self.c)
        y = y.reshape(B, self.c, H, W) + self.bias.view(1, -1, 1, 1)
        return self.act(self.bn(y)) + self.skip(x)


# ---------------------------------------------------------------- B. CSSAM
class CSSAMConcat(nn.Module):
    """Cross-Scale Semantic Alignment (Eq. 6-9) in the position of a neck Concat.

    Input  : [X_sem (upsampled semantic map), X_spa (backbone detail map)]
    Output : concat([X_sem, S * X_spa]);  S = Softplus(F_trans(M * X_spa)),
             M in R^{B x 1 x H x W}.
    """

    def __init__(self, c_sem, c_spa, c_a=64, beta=1.0):
        super().__init__()
        self.align = Conv(c_sem, c_a, 1)                 # A_theta
        self.proj = Conv(c_spa, c_a, 1)
        self.R = Conv(2 * c_a, c_a, 3)                   # cross-relation branch
        self.C = Conv(c_a, c_a, 1)                       # semantic branch
        self.D = Conv(c_a, c_a, 3, g=c_a)                # spatial branch
        self.H = nn.Conv2d(c_a, 1, 1)                    # combination -> M
        self.trans = nn.Conv2d(c_spa, c_spa, 3, padding=1, groups=c_spa)   # F_trans
        nn.init.normal_(self.trans.weight, std=0.01)
        nn.init.constant_(self.trans.bias, 0.5413)       # Softplus(0.5413) = 1
        self.beta = beta

    def forward(self, xs):
        x_sem, x_spa = xs
        sem = self.align(x_sem)
        if sem.shape[-2:] != x_spa.shape[-2:]:
            sem = F.interpolate(sem, size=x_spa.shape[-2:], mode="bilinear", align_corners=False)
        sp = self.proj(x_spa)
        M = self.H(self.R(torch.cat([sp, sem], 1)) + self.C(sem) + self.D(sp))
        S = F.softplus(self.trans(M * x_spa), beta=self.beta)
        return torch.cat([x_sem, S * x_spa], 1)


# ---------------------------------------------------------------- C. DFRM
class EMA(nn.Module):
    """Efficient Multi-Scale Attention (Ouyang et al., 2023)."""

    def __init__(self, channels, factor=8):
        super().__init__()
        self.g = factor
        assert channels % self.g == 0
        cg = channels // self.g
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1x1 = nn.Conv2d(cg, cg, 1)
        self.conv3x3 = nn.Conv2d(cg, cg, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        gx = x.reshape(b * self.g, -1, h, w)
        x_h = self.pool_h(gx)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], 2))
        x_h, x_w = torch.split(hw, [h, w], 2)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)
        x11 = self.softmax(self.agp(x1).reshape(b * self.g, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.g, c // self.g, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.g, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.g, c // self.g, -1)
        wgt = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.g, 1, h, w)
        return (gx * wgt.sigmoid()).reshape(b, c, h, w)


class SE(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018)."""

    def __init__(self, c, r=16):
        super().__init__()
        m = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, m), nn.ReLU(inplace=True), nn.Linear(m, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


class DFRMRefine(nn.Sequential):
    """EMA followed by SE."""

    def __init__(self, c):
        super().__init__(EMA(c), SE(c))


class DySample(nn.Module):
    """DySample, 'lp' style (Liu et al., ICCV 2023); Eq. 10-11."""

    def __init__(self, c, scale=2, groups=4):
        super().__init__()
        self.scale, self.groups = scale, groups
        assert c >= groups and c % groups == 0
        self.offset = nn.Conv2d(c, 2 * groups * scale ** 2, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2) \
            .repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos.to(x.dtype)
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        ch = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        cw = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid([cw, ch], indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        norm = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / norm - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale) \
            .view(B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        out = F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords.to(x.dtype),
                            mode="bilinear", align_corners=False, padding_mode="border")
        return out.view(B, -1, self.scale * H, self.scale * W)


# ------------------------------------------------------- insertion into YOLO11
def _inherit(new, old):
    for a in ("f", "i", "type", "np"):
        setattr(new, a, getattr(old, a))
    return new


class _Post(nn.Module):
    """Original layer followed by an extra block; keeps the YOLO routing attributes."""

    def __init__(self, layer, block):
        super().__init__()
        self.layer, self.block = layer, block
        _inherit(self, layer)

    def forward(self, x):
        return self.block(self.layer(x))


@torch.no_grad()
def _layer_channels(model, imgsz=256):
    chans, hooks = {}, []
    for idx, m in enumerate(model.model):
        hooks.append(m.register_forward_hook(
            lambda mod, inp, out, idx=idx: chans.__setitem__(idx, out.shape[1]) if torch.is_tensor(out) else None))
    was_training = model.training
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return chans


def apply_amsfn(model, use_a=True, use_b=True, use_c=True):
    """Insert ARFAM (A), CSSAM (B) and DFRM (C) into a YOLO11 DetectionModel in place."""
    seq = model.model
    ch = _layer_channels(model)
    names = [m.__class__.__name__ for m in seq]
    assert names[11] == "Upsample" and names[12] == "Concat" and names[15] == "Concat"

    if use_a:
        for i in (4, 6, 8):
            seq[i] = _Post(seq[i], ARFAM(ch[i]))
    if use_b:
        seq[12] = _inherit(CSSAMConcat(ch[11], ch[6]), seq[12])
        seq[15] = _inherit(CSSAMConcat(ch[14], ch[4]), seq[15])
    if use_c:
        for i in (11, 14):
            seq[i] = _inherit(DySample(ch[i - 1]), seq[i])
        for i in (16, 19, 22):
            seq[i] = _Post(seq[i], DFRMRefine(ch[i]))
    return model
