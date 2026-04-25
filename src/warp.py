import torch
import torch.nn.functional as F


def backward_warp(frame: torch.Tensor, flow: torch.Tensor):
    """
    Warp `frame` (LR t-1) to align with frame t using backward sampling.

    Args:
        frame : (B, C, H, W)  — the previous frame to sample from
        flow  : (B, 2)         — rigid pan offset in LR pixels [dx, dy]
                                 where dx = -(actual_offset_x / scale_factor)
                                       dy = -(actual_offset_y / scale_factor)
    Returns:
        warped : (B, C, H, W)
        mask   : (B, 1, H, W)  — 1 where sampling was in-bounds, 0 elsewhere
    """
    B, C, H, W = frame.shape
    device = frame.device

    # Base grid: normalized coords in [-1, 1] for every pixel in output
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    base_grid = torch.stack([grid_x, grid_y], dim=-1)       # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # Convert pixel-space flow to normalized-space flow
    dx = flow[:, 0]  # (B,)
    dy = flow[:, 1]  # (B,)
    dx_norm = (2.0 * dx / max(W - 1, 1)).view(B, 1, 1, 1)
    dy_norm = (2.0 * dy / max(H - 1, 1)).view(B, 1, 1, 1)

    flow_norm = torch.cat([dx_norm.expand(B, H, W, 1),
                           dy_norm.expand(B, H, W, 1)], dim=-1)  # (B, H, W, 2)

    sample_grid = base_grid + flow_norm  # backward sampling locations

    # Occlusion mask: valid where sampling is inside [-1, 1]
    in_x = (sample_grid[..., 0].abs() <= 1.0)
    in_y = (sample_grid[..., 1].abs() <= 1.0)
    mask = (in_x & in_y).float().unsqueeze(1)  # (B, 1, H, W)

    warped = F.grid_sample(
        frame, sample_grid,
        mode="bilinear", padding_mode="border", align_corners=True
    )

    return warped, mask
