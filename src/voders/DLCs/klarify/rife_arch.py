import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Tuple, Optional, Union

_backwarp_tenGrid = {}

def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    device = tenInput.device
    k = (str(device), str(tenFlow.size()))
    if k not in _backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        _backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)
    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
    ], 1)
    g = (_backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear',
                         padding_mode='border', align_corners=True)

def conv(in_planes: int, out_planes: int, kernel_size: int = 3,
         stride: int = 1, padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, feat: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3

class ResConv(nn.Module):
    def __init__(self, c: int, dilation: int = 1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock_Heavy(nn.Module):
    def __init__(self, in_planes: int, c: int = 64):
        super(IFBlock_Heavy, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor, flow: Optional[torch.Tensor] = None,
                scale: float = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                               align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow_out = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow_out, mask, feat

class IFNet_Heavy(nn.Module):
    def __init__(self):
        super(IFNet_Heavy, self).__init__()
        self.block0 = IFBlock_Heavy(7 + 8, c=192)
        self.block1 = IFBlock_Heavy(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock_Heavy(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock_Heavy(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock_Heavy(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(self, x: torch.Tensor, timestep: float = 0.5,
                scale_list: List[float] = [8, 4, 2, 1],
                training: bool = False, fastmode: bool = True,
                ensemble: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor, List]:
        if not training:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        else:
            img0 = x[:, :3]
            img1 = x[:, 3:6]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        feat = None
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            if flow is None:
                flow, mask, feat = blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None, scale=scale_list[i]
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = blocks[i](
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1,
                              timestep, mask, feat), 1),
                    flow, scale=scale_list[i]
                )
                mask = m0
                flow = flow + fd
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        merged[4] = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow_list, mask_list[4], merged

class IFBlock_Lite(nn.Module):
    def __init__(self, in_planes: int, c: int = 64):
        super(IFBlock_Lite, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor, flow: Optional[torch.Tensor] = None,
                scale: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                               align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow_out = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow_out, mask

class IFNet_Lite(nn.Module):
    def __init__(self):
        super(IFNet_Lite, self).__init__()
        self.block0 = IFBlock_Lite(7 + 8, c=128)
        self.block1 = IFBlock_Lite(8 + 4 + 8, c=96)
        self.block2 = IFBlock_Lite(8 + 4 + 8, c=64)
        self.block3 = IFBlock_Lite(8 + 4 + 8, c=48)
        self.encode = Head()

    def forward(self, x: torch.Tensor, timestep: float = 0.5,
                scale_list: List[float] = [8, 4, 2, 1],
                training: bool = False, fastmode: bool = True,
                ensemble: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor, List]:
        if not training:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        else:
            img0 = x[:, :3]
            img1 = x[:, 3:6]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        blocks = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None, scale=scale_list[i]
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0 = blocks[i](
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1,
                              timestep, mask), 1),
                    flow, scale=scale_list[i]
                )
                mask = m0
                flow = flow + fd
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        merged[3] = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow_list, mask_list[3], merged

class RIFE:
    def __init__(self, mode: str = 'heavy'):
        self.mode = mode
        if mode == 'heavy':
            self.flownet = IFNet_Heavy()
            self.version = 4.25
        else:
            self.flownet = IFNet_Lite()
            self.version = 4.17
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(self.device_type)

    def load_model(self, path: str, mode: str = None, rank: int = 0):
        if mode is None:
            mode = self.mode
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        model_filename = f'framegen-{mode}.pkl'
        model_path = os.path.join(path, model_filename)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            for k in list(state_dict.keys()):
                state_dict[k] = state_dict[k].float()
            converted = convert(state_dict)
            if len(converted) > 0:
                state_dict = converted
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('teacher') and not k.startswith('caltime'):
                    filtered_state_dict[k] = v
            self.flownet.load_state_dict(filtered_state_dict, strict=True)
            self.flownet.float()
            self.mode = mode
            self.version = 4.25 if mode == 'heavy' else 4.17
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    def save_model(self, path: str, rank: int = 0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), f'{path}/framegen-{self.mode}.pkl')

    def inference(self, img0: torch.Tensor, img1: torch.Tensor,
                  timestep: float = 0.5, scale: float = 1.0) -> torch.Tensor:
        imgs = torch.cat((img0, img1), 1)
        if self.mode == 'heavy':
            scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        else:
            scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]

Model = RIFE
