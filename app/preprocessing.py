"""
Volume loading and preprocessing.

Handles NIfTI (.nii/.nii.gz), NumPy (.npy/.npz), DICOM (.dcm),
TIFF (.tif/.tiff), and common 2D image formats (.png, .jpg, .jpeg, .bmp, .webp).

2D images (PNG, JPG, single-frame TIFF, single-slice DICOM) are automatically
converted to pseudo-3D volumes by replicating the slice along the depth axis so
the 3D model can process them.  Classification quality is lower for 2D inputs
than for full 3D NIfTI/NPY volumes.

Matches the preprocessing pipeline from fixing_size.py.
"""

import os
import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom

ORIG_SHAPE = (160, 192, 160)
CROP_SHAPE = (96, 112, 96)


# ============================================================
# Format-specific loaders
# ============================================================

def _load_image_2d(file_path):
    """Load a 2D bitmap image (PNG/JPG/BMP/WEBP) as grayscale float32."""
    from PIL import Image
    img = Image.open(file_path).convert('L')
    return np.array(img, dtype=np.float32)


def _load_tiff(file_path):
    """Load TIFF – single-frame returns 2D array, multi-frame returns 3D (H,W,D)."""
    from PIL import Image, ImageSequence
    img = Image.open(file_path)
    frames = [np.array(frame.convert('L'), dtype=np.float32)
              for frame in ImageSequence.Iterator(img)]
    if len(frames) == 1:
        return frames[0]          # 2D – caller will expand to 3D
    return np.stack(frames, axis=2)   # (H, W, D)


def _load_dicom(file_path):
    """Load a DICOM file as float32 array (2D single slice or 3D volume)."""
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required to load DICOM files. "
            "Install it with:  pip install pydicom"
        )
    dcm = pydicom.dcmread(file_path)
    data = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, 'RescaleSlope', 1.0))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
    data = data * slope + intercept
    return data


def _2d_to_3d_volume(image_2d):
    """Convert a 2D MRI slice to a pseudo-3D volume.

    The 2D image is resized to (ORIG_SHAPE[0], ORIG_SHAPE[1]) and then
    replicated ORIG_SHAPE[2] times along the depth axis.  This allows the
    3D model to process single-slice MRI images downloaded from the internet,
    at the cost of reduced accuracy compared to real 3D volumetric scans.

    Args:
        image_2d: 2D numpy array (H, W)

    Returns:
        3D numpy array (ORIG_SHAPE[0], ORIG_SHAPE[1], ORIG_SHAPE[2])
    """
    h, w = image_2d.shape
    th, tw = ORIG_SHAPE[0], ORIG_SHAPE[1]
    zoom_h = th / h if h > 0 else 1.0
    zoom_w = tw / w if w > 0 else 1.0
    resized = scipy_zoom(image_2d.astype(np.float32), (zoom_h, zoom_w), order=1)
    depth = ORIG_SHAPE[2]
    volume = np.repeat(resized[:, :, np.newaxis], depth, axis=2)
    return volume.astype(np.float32)


# ============================================================
# Core load function (internal – returns format metadata too)
# ============================================================

def _load_volume_internal(file_path):
    """Load a volume from any supported format.

    Returns:
        data: 3D numpy float32 array
        format_info: dict with keys
            'format'       – human-readable format name
            'is_2d_input'  – True if a 2D image was expanded to 3D
    """
    ext = _get_extension(file_path)
    format_info = {'format': ext, 'is_2d_input': False}

    # ---- NIfTI ----
    if ext in ('.nii', '.nii.gz'):
        import nibabel as nib
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data.mean(axis=3) if data.shape[3] <= 10 else data[:, :, :, 0]
        format_info['format'] = 'NIfTI'

    # ---- NumPy array ----
    elif ext == '.npy':
        data = np.load(file_path).astype(np.float32)
        format_info['format'] = 'NumPy (.npy)'

    # ---- Compressed NumPy ----
    elif ext == '.npz':
        npz = np.load(file_path)
        key = list(npz.keys())[0]
        data = npz[key].astype(np.float32)
        format_info['format'] = 'NumPy (.npz)'

    # ---- DICOM ----
    elif ext in ('.dcm', '.dicom', '.ima'):
        data = _load_dicom(file_path)
        format_info['format'] = 'DICOM'

    # ---- TIFF (single or multi-frame) ----
    elif ext in ('.tif', '.tiff'):
        data = _load_tiff(file_path)
        format_info['format'] = 'TIFF'

    # ---- Common 2D image formats ----
    elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
        data = _load_image_2d(file_path)
        format_info['format'] = f'2D Image ({ext})'
        format_info['is_2d_input'] = True

    else:
        raise ValueError(
            f"Unsupported file format: {ext!r}.\n"
            "Supported formats:\n"
            "  3D volumes : .nii, .nii.gz, .npy, .npz\n"
            "  DICOM      : .dcm\n"
            "  TIFF       : .tif, .tiff  (multi-frame = 3D)\n"
            "  2D images  : .png, .jpg, .jpeg, .bmp, .webp\n"
            "               (converted to pseudo-3D automatically)"
        )

    # 4-D handling (e.g. fMRI or multi-echo DICOM)
    if data.ndim == 4:
        data = data.mean(axis=3) if data.shape[3] <= 10 else data[:, :, :, 0]

    # 2-D → 3-D expansion
    if data.ndim == 2:
        format_info['is_2d_input'] = True
        data = _2d_to_3d_volume(data)

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume after loading, got shape {data.shape}"
        )

    return data, format_info


# ============================================================
# Public API
# ============================================================

def load_volume(file_path):
    """Load a brain volume from any supported file format.

    Supported formats:
      - NIfTI   : .nii, .nii.gz
      - NumPy   : .npy, .npz
      - DICOM   : .dcm  (requires pydicom)
      - TIFF    : .tif, .tiff  (multi-frame stacks treated as 3D)
      - 2D image: .png, .jpg, .jpeg, .bmp, .webp
                  (auto-expanded to pseudo-3D by slice replication)

    Args:
        file_path: path to file

    Returns:
        numpy array (H, W, D) as float32
    """
    data, _ = _load_volume_internal(file_path)
    return data


def normalize_volume(data, low_pct=1.0, high_pct=99.0):
    """Percentile normalization to [-1, 1].

    Matches fixing_size.py:264-278 preprocessing pipeline.

    Args:
        data: 3D numpy array
        low_pct: lower percentile for clipping
        high_pct: upper percentile for clipping

    Returns:
        normalized volume in [-1, 1]
    """
    data = data.copy()
    data[data < 0] = 0

    nonzero = data[data > 0]
    if len(nonzero) < 100:
        return data

    v_min = np.percentile(nonzero, low_pct)
    v_max = np.percentile(nonzero, high_pct)

    np.clip(data, v_min, v_max, out=data)
    data = (data - v_min) / (v_max - v_min + 1e-8)
    data = 2.0 * data - 1.0

    return data.astype(np.float32)


def _is_already_normalized(data):
    """Check if volume is already normalized to approximately [-1, 1]."""
    return data.min() >= -1.1 and data.max() <= 1.1


def resize_volume(volume, target_shape=ORIG_SHAPE):
    """Resize volume to target shape using trilinear interpolation.

    Args:
        volume: 3D numpy array
        target_shape: target (H, W, D)

    Returns:
        resized volume
    """
    if volume.shape == target_shape:
        return volume

    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return scipy_zoom(volume, zoom_factors, order=1).astype(np.float32)


def center_crop_3d(volume, target_shape=CROP_SHAPE):
    """Center crop a 3D volume to target shape, with padding if smaller.

    Args:
        volume: 3D numpy array
        target_shape: target (D, H, W)

    Returns:
        cropped volume
    """
    d, h, w = volume.shape
    td, th, tw = target_shape
    sd = max(0, (d - td) // 2)
    sh = max(0, (h - th) // 2)
    sw = max(0, (w - tw) // 2)
    cropped = volume[sd:sd + td, sh:sh + th, sw:sw + tw]

    if cropped.shape != target_shape:
        padded = np.full(target_shape, -1.0, dtype=np.float32)
        pd_ = (td - cropped.shape[0]) // 2
        ph_ = (th - cropped.shape[1]) // 2
        pw_ = (tw - cropped.shape[2]) // 2
        padded[pd_:pd_ + cropped.shape[0],
               ph_:ph_ + cropped.shape[1],
               pw_:pw_ + cropped.shape[2]] = cropped
        return padded
    return cropped


def preprocess_for_model(file_path, target_shape=ORIG_SHAPE, crop_shape=CROP_SHAPE):
    """Full preprocessing pipeline for an uploaded volume.

    Automatically handles all supported file formats including plain 2D images
    (PNG, JPG, etc.) downloaded from the internet.

    Steps:
    1. Load volume (any supported format; 2D images expanded to pseudo-3D)
    2. Normalize to [-1, 1] if not already normalized
    3. Resize to target_shape (160, 192, 160)
    4. Center crop to crop_shape (96, 112, 96)
    5. Convert to torch tensor (1, 1, D, H, W)

    Args:
        file_path: path to volume file
        target_shape: full volume shape
        crop_shape: cropped shape for model input

    Returns:
        tensor: (1, 1, D, H, W) float32 tensor for model input
        full_volume: (H, W, D) numpy array at target_shape (for visualization)
        format_info: dict with 'format' (str) and 'is_2d_input' (bool)
    """
    volume, format_info = _load_volume_internal(file_path)

    if not _is_already_normalized(volume):
        volume = normalize_volume(volume)

    volume = resize_volume(volume, target_shape)
    full_volume = volume.copy()

    cropped = center_crop_3d(volume, crop_shape)
    tensor = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    return tensor, full_volume, format_info


def _get_extension(file_path):
    """Get file extension, handling .nii.gz."""
    path = file_path.lower()
    if path.endswith('.nii.gz'):
        return '.nii.gz'
    return os.path.splitext(path)[1].lower()
