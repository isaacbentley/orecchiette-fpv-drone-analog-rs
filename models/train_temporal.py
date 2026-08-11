import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from torch.export import export, ExportedProgram

class TemporalDenoiser(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, num_layers=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        layers = []
        # Input layer: concatenates image (1) and hidden state (hidden_channels)
        layers.append(nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            
        # Output layer: produces image (1) and new hidden state (hidden_channels)
        layers.append(nn.Conv2d(hidden_channels, in_channels + hidden_channels, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, input, hidden_in):
        x = torch.cat([input, hidden_in], dim=1)
        out = self.net(x)
        
        image_out = out[:, :1, :, :]
        # Residual connection for the image part
        image_out = input + image_out
        
        hidden_out = out[:, 1:, :, :]
        return image_out, hidden_out

class AnalogNoiseDataset(Dataset):
    def __init__(self, base_dataset, seq_length=5, patch_size=128):
        self.base_dataset = base_dataset
        self.seq_length = seq_length
        self.patch_size = patch_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get clean image (C, H, W) in [0, 1]
        img, _ = self.base_dataset[idx]
        
        # Convert to grayscale if it's RGB
        if img.shape[0] == 3:
            img = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
            
        # Random crop to patch_size
        _, h, w = img.shape
        if h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            img = img[:, top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize if too small (simplification)
            img = transforms.functional.resize(img, [self.patch_size, self.patch_size])

        # Generate sequence
        clean_seq = img.unsqueeze(0).repeat(self.seq_length, 1, 1, 1) # (T, C, H, W)
        noisy_seq = torch.zeros_like(clean_seq)
        
        for t in range(self.seq_length):
            frame = clean_seq[t, 0].numpy()
            
            # 1. Baseband signal: Map 0-1 to FM deviation range (e.g. -1 to 1)
            # Simulate a continuous 1D analog signal (raster scan)
            baseband = frame.flatten() * 2.0 - 1.0 
            
            # 1.5 Add colour subcarrier to simulate dot crawl (NTSC/PAL)
            # W0 depends on pixel clock vs subcarrier freq. ~0.5*pi is typical.
            w0 = 0.53 * np.pi
            chroma_amp = np.random.rand() * 0.3  # random saturation
            chroma_phase = np.random.rand() * 2 * np.pi
            t_pixels = np.arange(len(baseband))
            subcarrier = chroma_amp * np.sin(w0 * t_pixels + chroma_phase)
            baseband = baseband + subcarrier
            # Delta t = 1/fs. Let's assume some arbitrary modulation index.
            mod_index = 2.0
            phase = np.cumsum(baseband * mod_index)
            
            # 3. Create complex RF signal
            rf = np.exp(1j * phase)
            
            # 4. RF Channel (AWGN and Slow Fading)
            snr_db = np.random.uniform(5.0, 20.0) # 5dB to 20dB SNR
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = 1.0 / snr_linear
            
            # Complex Gaussian Noise
            noise = np.sqrt(noise_power / 2) * (np.random.randn(*rf.shape) + 1j * np.random.randn(*rf.shape))
            
            # Fading (Multipath / Antenna blockage)
            fade = 1.0 - np.random.rand() * 0.5
            
            rf_rx = rf * fade + noise
            
            # 5. FM Demodulation
            # angle difference between consecutive samples (d/dt phase)
            phase_rx = np.unwrap(np.angle(rf_rx))
            demod = np.diff(phase_rx, prepend=phase_rx[0]) / mod_index
            
            # 6. Map back to 0-1 and reshape
            noisy_frame = (demod + 1.0) / 2.0
            noisy_frame = np.clip(noisy_frame, 0.0, 1.0).reshape(frame.shape)
            
            noisy_seq[t, 0] = torch.from_numpy(noisy_frame).float()
            
        return noisy_seq, clean_seq

def export_and_quantize(model, hidden_channels=16):
    model.eval()
    
    # Dummy inputs for export
    dummy_input = torch.randn(1, 1, 128, 128)
    dummy_hidden = torch.zeros(1, hidden_channels, 128, 128)
    
    fp32_path = "models/temporal_trained_fp32.onnx"
    quant_path = "models/temporal_quantized_trained.onnx"
    
    print(f"\nExporting model to {fp32_path}...")
    torch.onnx.export(
        model, 
        (dummy_input, dummy_hidden),
        fp32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input', 'hidden_in'],
        output_names=['output', 'hidden_out'],
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},
            'hidden_in': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'},
            'hidden_out': {2: 'height', 3: 'width'}
        }
    )
    
    print("Quantizing to INT8...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    quantize_dynamic(
        model_input=fp32_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8
    )
    print(f"Saved INT8 ONNX model to {quant_path}")
    
    # Clean up FP32 model and its external data if any
    if os.path.exists(fp32_path):
        os.remove(fp32_path)
    if os.path.exists(fp32_path + ".data"):
        os.remove(fp32_path + ".data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dummy-data", action="store_true", help="Use a tiny random dataset for sanity checking")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Base Dataset
    transform = transforms.ToTensor()
    if args.dummy_data:
        print("Using dummy random data...")
        # A list of random image tensors simulating a dataset
        base_dataset = [(torch.rand(3, 128, 128), 0) for _ in range(32)]
    else:
        print("Downloading/Loading STL10 Dataset...")
        base_dataset = torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=transform)

    train_dataset = AnalogNoiseDataset(base_dataset, seq_length=args.seq_length, patch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = TemporalDenoiser(in_channels=1, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (noisy_seq, clean_seq) in enumerate(train_loader):
            noisy_seq, clean_seq = noisy_seq.to(device), clean_seq.to(device) # (B, T, C, H, W)
            B, T, C, H, W = noisy_seq.shape
            
            optimizer.zero_grad()
            
            hidden = torch.zeros(B, args.hidden_channels, H, W, device=device)
            loss = 0.0
            
            for t in range(T):
                img_in = noisy_seq[:, t]
                target = clean_seq[:, t]
                
                img_out, hidden = model(img_in, hidden)
                loss += criterion(img_out, target)
                
            loss = loss / T
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        print(f"==> Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")

    print("Training complete.")
    export_and_quantize(model, hidden_channels=args.hidden_channels)

if __name__ == "__main__":
    main()
