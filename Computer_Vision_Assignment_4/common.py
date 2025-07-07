"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        c3_channels = dummy_out["c3"].shape[1]
        c4_channels = dummy_out["c4"].shape[1]
        c5_channels = dummy_out["c5"].shape[1]

        self.fpn_params["lat_c3"] = nn.Conv2d(c3_channels, self.out_channels, kernel_size=1)
        self.fpn_params["lat_c4"] = nn.Conv2d(c4_channels, self.out_channels, kernel_size=1)
        self.fpn_params["lat_c5"] = nn.Conv2d(c5_channels, self.out_channels, kernel_size=1)

        self.fpn_params["out_p3"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_params["out_p4"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.fpn_params["out_p5"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        c3, c4, c5 = backbone_feats["c3"], backbone_feats["c4"], backbone_feats["c5"]
        lat_c3 = self.fpn_params["lat_c3"](c3)
        lat_c4 = self.fpn_params["lat_c4"](c4)
        lat_c5 = self.fpn_params["lat_c5"](c5)

        p5 = self.fpn_params["out_p5"](lat_c5)

        p4 = lat_c4 + F.interpolate(lat_c5, size=lat_c4.shape[-2:], mode="nearest")
        p4 = self.fpn_params["out_p4"](p4)

        p3 = lat_c3 + F.interpolate(p4, size=lat_c3.shape[-2:], mode="nearest")
        p3 = self.fpn_params["out_p3"](p3)

        fpn_feats["p3"] = p3
        fpn_feats["p4"] = p4
        fpn_feats["p5"] = p5
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        B, C, H, W = feat_shape
        stride = strides_per_fpn_level[level_name]

        shifts_x = torch.arange(0, W, dtype=dtype, device=device) + 0.5  
        shifts_y = torch.arange(0, H, dtype=dtype, device=device) + 0.5

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        xc = shift_x * stride  
        yc = shift_y * stride  

        coords = torch.stack([xc, yc], dim=-1).reshape(-1, 2)

        location_coords[level_name] = coords
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

        # ------------------------------------------------------------------
    # Pure-PyTorch Non-Maximum Suppression (single class).
    # Works on both CPU and CUDA tensors because it uses only tensor ops.
    # ------------------------------------------------------------------
    # Short-cuts: if there are no boxes or scores, return empty tensor.
    if boxes.numel() == 0 or scores.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Unpack coordinates and pre-compute the area of each box
    x1, y1, x2, y2 = boxes.t()                       # shape (N,)
    areas = (x2 - x1) * (y2 - y1)                    # shape (N,)

    # Sort by score (highest first)
    order = scores.argsort(descending=True)          # shape (N,)
    keep_indices = []                                # list of ints

    while order.numel() > 0:
        i = order[0]                                 # current best box
        keep_indices.append(i.item())

        if order.numel() == 1:                       # nothing left to compare
            break

        # Compute IoU of the best box with the rest
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter   = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - inter
        iou   = inter / union                       # shape (len(order)-1,)

        # Keep boxes with IoU <= threshold and iterate again
        remaining = iou <= iou_threshold
        order = order[1:][remaining]

    # Convert Python list to a torch.LongTensor on the same device
    keep = torch.as_tensor(keep_indices,
                           dtype=torch.long,
                           device=boxes.device)
    return keep



def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
