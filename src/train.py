import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SRTemporalDataset, download_sample_images
from model import TemporalSRResNet
from tqdm import tqdm

def train(num_epochs=10, batch_size=2, lr=1e-4, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "data/samples"
    download_sample_images(output_dir=data_dir)

    # Use the new Temporal Dataset
    dataset = SRTemporalDataset(data_dir, scale_factor=4, crop_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Use the new Temporal Model
    model = TemporalSRResNet(scale_factor=4).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {num_epochs} epochs...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # The dataset now returns 3 items: prev_lr, curr_lr, curr_hr
        for batch_lr_prev, batch_lr_curr, batch_hr_curr in pbar:
            batch_lr_prev = batch_lr_prev.to(device)
            batch_lr_curr = batch_lr_curr.to(device)
            batch_hr_curr = batch_hr_curr.to(device)

            # Forward pass: pass both frames to the model
            outputs = model(batch_lr_prev, batch_lr_curr)

            # Compute loss against the current HR frame
            loss = criterion(outputs, batch_hr_curr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save checkpoints
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(save_dir, f"tsr_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Temporal Super-Resolution Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    args = parser.parse_args()

    train(num_epochs=args.epochs, batch_size=args.batch_size)
