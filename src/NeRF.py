import jittor as jt
import numpy as np
from jittor import nn
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_x_channels=3, input_d_channels=3, output_channels=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_x_channels = input_x_channels
        self.input_d_channels = input_d_channels
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_x_channels, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_x_channels, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_d_channels + W, W//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_channels)

    def execute(self, x):
        input_pts, input_views = jt.split(x, [self.input_x_channels, self.input_d_channels], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jt.nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = nn.relu(h)

            rgb = self.rgb_linear(h)
            outputs = jt.concat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    