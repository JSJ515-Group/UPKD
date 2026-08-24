"""Cross-stage feature transfer module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTransfer(nn.Module):
    """Transfer and merge features across adjacent network stages.

    The current-stage feature is first projected into an intermediate feature
    space. When a deeper-stage feature is provided, the two features are
    resized and merged with learned element-wise weights. The merged feature is
    then mapped to the target channel dimension.
    """

    def __init__(self,in_channel: int,mid_channel: int,out_channel: int,fuse: bool):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )

        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """Return the transferred output and the intermediate feature.

        Args:
            x: Current-stage feature with shape ``[B, C, H, W]``.
            y: Feature transferred from a deeper stage. It is required when
                feature fusion is enabled.

        Returns:
            A tuple containing the output feature and the intermediate feature
            passed to the next shallower stage.
        """
        n, _, h, w = x.shape
        x = self.conv1(x)

        if self.att_conv is not None:
            if y is None:
                raise ValueError(
                    "The transferred feature must not be None when fusion is enabled."
                )

            if y.shape[-2:] != (h, w):
                y = F.interpolate(y, size=(h, w), mode="nearest")

            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)

        out = self.conv2(x)
        return out, x
