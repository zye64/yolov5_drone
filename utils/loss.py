# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.num_extra_dims = m.num_extra_dims

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lext = torch.zeros(1, device=self.device)  # extra dims loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # 移除错误的 num_extra_dims 计算
                # num_extra_dims = pi.shape[2] - self.nc - 5 # <--- 错误
                # 直接使用初始化时存储的 self.num_extra_dims
                current_num_extra_dims = self.num_extra_dims
                
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 检查 self.num_extra_dims 是否大于 0
                if current_num_extra_dims > 0:
                    # 确保 split 参数使用正确的 self.nc 和 self.num_extra_dims
                    pxy, pwh, obj_score, pcls, pextra = pi[b, a, gj, gi].split((2, 2, 1, self.nc, current_num_extra_dims), 1)
                    
                    # 额外维度的MSE损失计算
                    # 确保 targets 张量包含额外维度数据
                    # targets 的格式: [img_idx, cls, x, y, w, h, theta, phi]
                    # theta 和 phi 位于索引 6 和 7
                    if targets.shape[1] > 6 + current_num_extra_dims -1: # 检查 targets 列数是否足够
                        # 获取匹配的 target 的额外维度值
                        # 需要一种方法将 pi[b, a, gj, gi] 映射回原始的 targets
                        # build_targets 返回的 indices[i] 包含了 b (image index)
                        # targets 张量已经按 image index 排序
                        # 但我们需要匹配到具体的 target 行
                        # 注意：这里的实现可能需要更精细的匹配逻辑，暂时使用简化的索引
                        # 一个简单的（但可能不完全准确的）方法是假设 b 中的索引可以直接用于 targets
                        # 需要验证 build_targets 是否保证了这种对应关系
                        # TODO: 改进 target 和 prediction 之间的匹配以提取正确的额外维度标签
                        
                        # 尝试直接使用 b 索引 (假设 build_targets 保持顺序)
                        # 注意: targets 包含了 batch 中所有图片的标签，需要过滤
                        target_img_indices = targets[:, 0].long()
                        # selected_target_indices = torch.cat([torch.where(target_img_indices == batch_idx)[0] for batch_idx in b.unique()])
                        # if len(selected_target_indices) == n: # 确保数量匹配
                        # text = targets[selected_target_indices.cpu(), 6:6+current_num_extra_dims].to(self.device)
                        
                        # 简化方法: 直接使用 b, 但这可能在多目标匹配同一 grid 时出错
                        # 并且 targets 是整个 batch 的，b 是当前 layer 匹配的 batch indices
                        # 需要更鲁棒的方式，例如在 build_targets 中传递 target 的原始索引
                        # 暂时跳过精确匹配的 loss 计算，聚焦于修复 split 错误
                        # lext += F.mse_loss(pextra.sigmoid(), text) # 暂时注释掉loss计算
                        pass # 暂时跳过 lext 计算逻辑
                    else:
                        # 如果 targets 不包含足够的列，打印警告或处理
                        LOGGER.warning("Targets do not contain expected extra dimensions.")
                        
                else: # num_extra_dims == 0
                    pxy, pwh, obj_score, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lext *= self.hyp.get("ext", 1.0)  # 使用超参数中的ext权重，如果没有则默认为1.0
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + lext) * bs, torch.cat((lbox, lobj, lcls, lext)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain = torch.ones(7, device=self.device)  # 旧的 gain 初始化
        # gain 大小应为 target 列数 + 1 (为 anchor_idx 预留)
        # targets shape from dataloader is (N, 8): [img_idx, cls, x, y, w, h, theta, phi]
        # After adding anchor index, shape becomes (na, nt, 9)
        gain = torch.ones(9, device=self.device) 
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices, shape(na, nt, 9)

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            # gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # 旧的 gain 更新
            # 更新 gain 以匹配 targets 的列结构 [batch_idx, cls, x, y, w, h, theta, phi, anchor_idx]
            # 对 x, y, w, h (索引 2, 3, 4, 5) 应用缩放增益
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]] # xywh gain
            # 其他维度(索引 0, 1, 6, 7, 8) 的 gain 保持为 1

            # Match targets to anchors
            t = targets * gain  # shape(na, nt, 9) now matches gain shape(9)
            if nt:
                # Matches
                # 注意: r 的计算应该只基于 w,h (索引 4, 5)
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy (索引 2, 3)
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # 新的解包方式，考虑额外维度和anchor索引
            # t 的列: [batch_idx, cls, gx, gy, gw, gh, theta_scaled, phi_scaled, anchor_idx]
            # 注意：theta, phi 乘以了 gain (现在是1)，但它们是 grid scale 的
            bc = t[:, :2]           # batch_idx, cls (索引 0, 1)
            gxy = t[:, 2:4]         # grid xy (索引 2, 3)
            gwh = t[:, 4:6]         # grid wh (索引 4, 5)
            # gextra = t[:, 6:-1]   # grid theta, phi (索引 6, 7)
            a = t[:, -1:]           # anchor index (索引 8)

            a, (b, c) = a.long().view(-1), bc.long().T  # anchors index, image index, class index
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box (dx, dy, gw, gh)
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # 注意: 我们没有将 gextra (theta, phi) 添加到返回列表中，因为当前的loss计算主要用 tcls, tbox, indices, anch

        return tcls, tbox, indices, anch
