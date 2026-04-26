import torch
import torch.nn.functional as F


def backward_warp(frame: torch.Tensor, flow: torch.Tensor):
    """
    Backward warp `frame` using a flow field.

    Supports two flow formats:
      Rigid  — flow: (B, 2)         constant displacement [dx, dy] in pixels
               used during synthetic training (known pan offset)
      Dense  — flow: (B, 2, H, W)   per-pixel displacement from optical flow
               used for real video (Farneback output)

    Convention: backward flow — for each pixel (i,j) in the output,
      sample `frame` at position (j + flow_x, i + flow_y).

    Args:
        frame : (B, C, H, W)
        flow  : (B, 2) or (B, 2, H, W)  — pixel-space backward displacement

    Returns:
        warped : (B, C, H, W)
        mask   : (B, 1, H, W)  — 1 where sampling was in-bounds, 0 elsewhere
    """
    B, C, H, W = frame.shape
    device = frame.device

    # Base sampling grid in normalised [-1, 1] coords
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
    base_grid = torch.stack([grid_x, grid_y], dim=-1)          # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)   # (B, H, W, 2)

    if flow.dim() == 2:
        # --- Rigid flow (B, 2) ---
        dx = flow[:, 0].view(B, 1, 1, 1)
        dy = flow[:, 1].view(B, 1, 1, 1)
        dx_norm = (2.0 * dx / max(W - 1, 1)).expand(B, H, W, 1)
        dy_norm = (2.0 * dy / max(H - 1, 1)).expand(B, H, W, 1)
        flow_norm = torch.cat([dx_norm, dy_norm], dim=-1)          # (B, H, W, 2)

    else:
        # --- Dense flow (B, 2, H, W) ---
        dx = flow[:, 0]   # (B, H, W)
        dy = flow[:, 1]   # (B, H, W)
        dx_norm = (2.0 * dx / max(W - 1, 1)).unsqueeze(-1)        # (B, H, W, 1)
        dy_norm = (2.0 * dy / max(H - 1, 1)).unsqueeze(-1)        # (B, H, W, 1)
        flow_norm = torch.cat([dx_norm, dy_norm], dim=-1)          # (B, H, W, 2)

    sample_grid = base_grid + flow_norm

    # Occlusion mask: pixels that sampled from outside the frame boundary
    in_x = sample_grid[..., 0].abs() <= 1.0
    in_y = sample_grid[..., 1].abs() <= 1.0
    mask  = (in_x & in_y).float().unsqueeze(1)                    # (B, 1, H, W)

    warped = F.grid_sample(
        frame, sample_grid,
        mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped, mask
