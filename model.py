import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetColorization(nn.Module):
    def __init__(self):
        super(UNetColorization, self).__init__()

        # Encoder
        self.enc1 = self.double_conv(1, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 512)

        # Decoder
        self.dec4 = self.up_conv(1024, 256)
        self.dec3 = self.up_conv(512, 128)
        self.dec2 = self.up_conv(256, 64)
        self.dec1 = self.up_conv(128, 64)

        # Final layer
        self.final = nn.Conv2d(64, 2, kernel_size=1)  # Output: 2 channels (ab)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.max_pool(e1))
        e3 = self.enc3(self.max_pool(e2))
        e4 = self.enc4(self.max_pool(e3))

        # Bottleneck
        b = self.bottleneck(self.max_pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat((self.crop_tensor(e4, b), b), dim=1))
        d3 = self.dec3(torch.cat((self.crop_tensor(e3, d4), d4), dim=1))
        d2 = self.dec2(torch.cat((self.crop_tensor(e2, d3), d3), dim=1))
        d1 = self.dec1(torch.cat((self.crop_tensor(e1, d2), d2), dim=1))

        # Final output
        out = self.final(d1)
        return torch.tanh(out)  # Output scaled to [-1, 1]

    @staticmethod
    def double_conv(in_channels, out_channels):
        """Two convolutional layers followed by ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def up_conv(in_channels, out_channels):
        """Upsampling followed by a double conv."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def max_pool(x):
        """2D max pooling."""
        return F.max_pool2d(x, kernel_size=2, stride=2)

    @staticmethod
    def crop_tensor(encoder_tensor, decoder_tensor):
        """Crop encoder tensor to match decoder tensor size."""
        _, _, h, w = decoder_tensor.size()
        return encoder_tensor[:, :, :h, :w]
    
    import torch
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage import io
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def load_model(model_path, device):
    model = UNetColorization().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def colorize_image(model, image_path, output_path, device='cpu'):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img_resized = img.resize((256, 256))

    # Create grayscale input
    gray_input = rgb2gray(np.array(img))
    input_image = Image.fromarray((gray_input * 255).astype(np.uint8))

    # Convert to Lab color space
    img_lab = rgb2lab(np.array(img_resized)).astype("float32")
    L = img_lab[:, :, 0] / 50.0 - 1.0
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)

    # Model inference
    with torch.no_grad():
        ab_pred = model(L_tensor)
        ab_pred = ab_pred.cpu().squeeze(0).numpy().transpose(1, 2, 0)

    # Process predicted ab channels
    ab_pred = ab_pred * 128.0
    ab_pred_uint8 = (ab_pred + 128).astype(np.uint8)
    ab_pred_resized = np.array(Image.fromarray(ab_pred_uint8).resize(original_size, Image.BILINEAR))
    ab_pred_resized = ab_pred_resized.astype(np.float32) - 128

    # Combine channels and convert to RGB
    img_lab_original = rgb2lab(np.array(img)).astype("float32")
    L_original = img_lab_original[:, :, 0]
    lab_pred = np.concatenate((L_original[:, :, np.newaxis], ab_pred_resized), axis=2)
    rgb_pred = lab2rgb(lab_pred)

    # Create output image
    output_image = Image.fromarray((np.clip(rgb_pred, 0, 1) * 255).astype(np.uint8))

    # Save individual output
    output_image.save(output_path)

    # Create comparison figure
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(input_image, cmap='gray')
    plt.title('Input (Grayscale)')
    plt.axis('off')


    plt.subplot(133)
    plt.imshow(output_image)
    plt.title('Colorized Output')
    plt.axis('off')

    # Save comparison first
    comparison_path = output_path.replace('.jpg', '_comparison.jpg')
    plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0, dpi=300)

    # Then display
    plt.show()

    # Close figure to free memory
    plt.close()

    return output_path, comparison_path