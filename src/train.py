import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SRDataset, download_sample_images
from model import SRResNet
from tqdm import tqdm

def train(num_epochs=10, batch_size=2, lr=1e-4, save_dir="checkpoints"):
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    data_dir = "data/samples"
    download_sample_images(output_dir=data_dir)

    # In a real scenario, you would have a much larger dataset.
    # Here we use our small sample dataset and set drop_last=False.
    dataset = SRDataset(data_dir, scale_factor=4, crop_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model, Loss, and Optimizer
    model = SRResNet(scale_factor=4).to(device)

    # Mean Squared Error (L2 Loss) or L1 Loss are standard for SR
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    print(f"Starting training for {num_epochs} epochs...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Use tqdm for a nice progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_lr, batch_hr in pbar:
            batch_lr = batch_lr.to(device)
            batch_hr = batch_hr.to(device)

            # Forward pass
            outputs = model(batch_lr)

            # Compute loss
            loss = criterion(outputs, batch_hr)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save checkpoint periodically or at the end
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(save_dir, f"sr_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train(num_epochs=5, batch_size=2)
