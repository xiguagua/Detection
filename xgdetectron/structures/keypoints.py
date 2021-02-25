import numpy as np
import torch

from detectron2.structures import Boxes


def gaussian_radius(boxes: Boxes, min_overlap=0.7) -> torch.Tensor:
    """[summary]

    Args:
        boxes (Boxes): Nx4 
        min_overlap (float, optional): [description]. Defaults to 0.7.

    Returns:
        radius: Nx1
    See: https://zhuanlan.zhihu.com/p/96856635
    """
    height, width = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return torch.min(
        torch.stack([r1, r2, r3], dim=1),
        dim=1,
        keepdim=True,
    ).values

def gaussian2D(shape, sigma=1):
    m, n = [((ss - 1.) / 2.).item() for ss in shape]
    # y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = torch.arange(-m, m+1).unsqueeze(1), torch.arange(-n, n+1).unsqueeze(0)
    y, x = torch.meshgrid(torch.arange(-m, m+1), torch.arange(-n, n+1))

    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
        
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: 
        # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = torch.tensor(value, dtype=torch.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = torch.ones((dim, diameter*2+1, diameter*2+1), dtype=torch.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)
    
    x, y = center

    height, width = heatmap.shape[0:2]
        
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom,
                             x - left:x + right]
    masked_regmap = regmap[:,
                           y - top:y + bottom,
                           x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                               radius - left:radius + right]
    masked_reg = reg[:,
                     radius - top:radius + bottom,
                     radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: 
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1]
        )
        masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap
