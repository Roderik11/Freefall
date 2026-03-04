"""
Extract alpha from PSD files.
Photoshop stores the alpha as a "spot channel" in the layer's channel data.
We read it from the flattened layer's channel with ChannelID.USER_LAYER_MASK or
as an extra channel beyond RGB.
"""
import os
import struct
import numpy as np
from PIL import Image

ASSETS_DIR = r"D:\Projects\2024\Freefall-Project\Assets"

PSD_FILES = [
    (r"Plants\Leaves\Textures\New_Leaves_Dif.psd", "New_Leaves_Dif.png"),
    (r"Wood\UnderBrush\Textures\UnderBrush.PSD", "UnderBrush.png"),
    (r"Plants\Root\Textures\Root_Dif.psd", "Root_Dif.png"),
]


def read_psd_with_alpha(path):
    """Read PSD and try to extract the extra alpha channel manually."""
    with open(path, 'rb') as f:
        # File Header
        sig = f.read(4)
        assert sig == b'8BPS', f"Not a PSD file: {sig}"
        version = struct.unpack('>H', f.read(2))[0]
        f.read(6)  # reserved
        channels = struct.unpack('>H', f.read(2))[0]
        height = struct.unpack('>I', f.read(4))[0]
        width = struct.unpack('>I', f.read(4))[0]
        depth = struct.unpack('>H', f.read(2))[0]
        color_mode = struct.unpack('>H', f.read(2))[0]
        print(f"  Header: {width}x{height}, {channels} channels, {depth}bpc, mode={color_mode}")

        # Color Mode Data
        cm_len = struct.unpack('>I', f.read(4))[0]
        f.read(cm_len)

        # Image Resources
        ir_len = struct.unpack('>I', f.read(4))[0]
        f.read(ir_len)

        # Layer and Mask Information
        lm_len = struct.unpack('>I', f.read(4))[0]
        f.seek(f.tell() + lm_len)  # skip layer/mask data

        # Image Data section — this is the merged/flattened image
        compression = struct.unpack('>H', f.read(2))[0]
        print(f"  Image data compression: {compression}")

        plane_size = width * height
        pixels = None

        if compression == 0:
            # Raw data
            data = f.read(channels * plane_size * (depth // 8))
            pixels = np.frombuffer(data, dtype=np.uint8 if depth == 8 else np.uint16)
            if depth == 16:
                pixels = (pixels.astype(np.float32) / 65535 * 255).astype(np.uint8)
            pixels = pixels.reshape(channels, height, width)

        elif compression == 1:
            # RLE: row byte counts for each channel, then RLE data
            # Read all row byte counts first
            row_counts = []
            for c in range(channels):
                counts = []
                for r in range(height):
                    counts.append(struct.unpack('>H', f.read(2))[0])
                row_counts.append(counts)

            # Decode each channel
            pixels = np.zeros((channels, height, width), dtype=np.uint8)
            for c in range(channels):
                for r in range(height):
                    row_data = f.read(row_counts[c][r])
                    # PackBits decompression
                    row_pixels = []
                    i = 0
                    while i < len(row_data) and len(row_pixels) < width:
                        n = row_data[i]
                        if n < 128:
                            # Copy next n+1 bytes
                            count = n + 1
                            row_pixels.extend(row_data[i+1:i+1+count])
                            i += 1 + count
                        elif n > 128:
                            # Repeat next byte 257-n times
                            count = 257 - n
                            row_pixels.extend([row_data[i+1]] * count)
                            i += 2
                        else:
                            # n == 128: no-op
                            i += 1
                    pixels[c, r, :] = np.array(row_pixels[:width], dtype=np.uint8)
        else:
            print(f"  Unsupported compression: {compression}")
            return None

        return pixels, width, height, channels


for psd_rel, png_name in PSD_FILES:
    psd_path = os.path.join(ASSETS_DIR, psd_rel)
    png_path = os.path.join(os.path.dirname(psd_path), png_name)

    if not os.path.exists(psd_path):
        print(f"NOT FOUND: {psd_rel}")
        continue

    print(f"\n=== {psd_rel} ===")
    result = read_psd_with_alpha(psd_path)

    if result is not None:
        pixels, w, h, ch = result
        rgb = np.stack([pixels[0], pixels[1], pixels[2]], axis=-1)

        if ch >= 4:
            alpha = pixels[3]
            print(f"  Alpha channel: min={alpha.min()} max={alpha.max()}")
        else:
            print("  No alpha channel. Deriving from luminance...")
            lum = rgb[:,:,0].astype(float)*0.299 + rgb[:,:,1].astype(float)*0.587 + rgb[:,:,2].astype(float)*0.114
            alpha = np.clip(lum * 3.0, 0, 255).astype(np.uint8)

        transparent_pct = (alpha < 128).sum() / alpha.size * 100
        print(f"  Final alpha: transparent={transparent_pct:.1f}%")

        rgba = np.dstack([rgb, alpha])
        img = Image.fromarray(rgba, 'RGBA')
        img.save(png_path, "PNG")
        print(f"  SAVED: {png_path} ({w}x{h})")
    else:
        print("  FAILED to read PSD")

print("\nDone!")
