import os
import random
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# =========================================================
# 1. Configuration
# =========================================================
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FIXED_FRAMES = 128
EPS = 1e-8


# =========================================================
# 2. Utility
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# 3. Audio augmentation helpers
# =========================================================
def trim_silence(
    y: np.ndarray,
    top_db: int = 30
) -> np.ndarray:
    """
    Remove leading/trailing silence.
    """
    if y is None or len(y) == 0:
        return y.astype(np.float32)

    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if y_trimmed is None or len(y_trimmed) == 0:
        return y.astype(np.float32)

    return y_trimmed.astype(np.float32)


def add_gaussian_noise(
    y: np.ndarray,
    noise_factor: float = 0.003
) -> np.ndarray:
    noise = np.random.randn(len(y)).astype(np.float32)
    y_noisy = y + noise_factor * noise
    return y_noisy.astype(np.float32)


def apply_gain(
    y: np.ndarray,
    gain_range: Tuple[float, float] = (0.9, 1.1)
) -> np.ndarray:
    gain = np.random.uniform(gain_range[0], gain_range[1])
    return (y * gain).astype(np.float32)


def time_stretch_audio(
    y: np.ndarray,
    rate_range: Tuple[float, float] = (0.9, 1.1)
) -> np.ndarray:
    rate = np.random.uniform(rate_range[0], rate_range[1])
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    return y_stretched.astype(np.float32)


def pitch_shift_audio(
    y: np.ndarray,
    sr: int,
    n_steps_range: Tuple[float, float] = (-1.5, 1.5)
) -> np.ndarray:
    n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y_shifted.astype(np.float32)


def normalize_amplitude(y: np.ndarray) -> np.ndarray:
    """
    Safe amplitude normalization.
    """
    if y is None or len(y) == 0:
        raise ValueError("Empty waveform.")

    max_val = np.max(np.abs(y))
    if max_val < EPS:
        return y.astype(np.float32)

    return (y / (max_val + EPS)).astype(np.float32)


def augment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    prob_noise: float = 0.4,
    prob_stretch: float = 0.3,
    prob_pitch: float = 0.3,
    prob_gain: float = 0.3
) -> np.ndarray:
    """
    Apply random augmentations to simulate domain shift.
    """
    y_aug = y.copy()

    if np.random.rand() < prob_noise:
        y_aug = add_gaussian_noise(y_aug, noise_factor=np.random.uniform(0.001, 0.006))

    if np.random.rand() < prob_stretch:
        y_aug = time_stretch_audio(y_aug, rate_range=(0.92, 1.08))

    if np.random.rand() < prob_pitch:
        y_aug = pitch_shift_audio(y_aug, sr=sr, n_steps_range=(-1.0, 1.0))

    if np.random.rand() < prob_gain:
        y_aug = apply_gain(y_aug, gain_range=(0.9, 1.1))

    y_aug = normalize_amplitude(y_aug)
    return y_aug.astype(np.float32)


# =========================================================
# 4. Load audio
# =========================================================
def load_audio(
    file_path: str,
    sr: int = SAMPLE_RATE,
    trim: bool = True,
    trim_top_db: int = 30
) -> np.ndarray:
    """
    Load audio as mono waveform with a fixed sample rate.
    Apply optional silence trimming and safe normalization.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    if y is None or len(y) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    if trim:
        y = trim_silence(y, top_db=trim_top_db)

    y = normalize_amplitude(y)
    return y.astype(np.float32)


# =========================================================
# 5. Extract log-mel spectrogram
# =========================================================
def extract_logmel(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fixed_frames: int = FIXED_FRAMES
) -> np.ndarray:
    """
    Convert waveform into fixed-size log-mel spectrogram.
    Output shape: (n_mels, fixed_frames)
    """
    if y is None or len(y) == 0:
        raise ValueError("Input waveform is empty.")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    current_frames = log_mel.shape[1]
    if current_frames < fixed_frames:
        pad_width = fixed_frames - current_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        log_mel = log_mel[:, :fixed_frames]

    return log_mel.astype(np.float32)


# =========================================================
# 6. Parse label / domain / section
# =========================================================
def parse_label(filename: str) -> str:
    name = filename.lower()

    if "anomaly" in name:
        return "anomaly"
    if "normal" in name:
        return "normal"
    return "unknown"


def parse_domain(filename: str) -> str:
    name = filename.lower()

    if "target" in name:
        return "target"
    if "source" in name:
        return "source"
    return "unknown"


def parse_section(filename: str) -> str:
    name = filename.lower()
    parts = name.replace("-", "_").split("_")

    for i, part in enumerate(parts):
        if part == "section" and i + 1 < len(parts):
            return f"section_{parts[i + 1]}"
        if part.startswith("section"):
            return part

    return "unknown"


# =========================================================
# 7. Build metadata table
# =========================================================
def build_metadata(
    data_root: str,
    machine_type: str = "fan",
    splits: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Expected structure:
    data_root/
        fan/
            train/
            test/
    """
    if splits is None:
        splits = ["train", "test"]

    rows = []

    for split in splits:
        split_dir = os.path.join(data_root, machine_type, split)

        if not os.path.exists(split_dir):
            print(f"Warning: folder not found -> {split_dir}")
            continue

        for fname in os.listdir(split_dir):
            if fname.lower().endswith(".wav"):
                file_path = os.path.join(split_dir, fname)
                rows.append({
                    "filepath": file_path,
                    "filename": fname,
                    "split": split,
                    "label": parse_label(fname),
                    "domain": parse_domain(fname),
                    "section": parse_section(fname),
                    "machine_type": machine_type
                })

    metadata_df = pd.DataFrame(rows)

    if metadata_df.empty:
        print(f"Warning: no .wav files found under {os.path.join(data_root, machine_type)}")

    return metadata_df


# =========================================================
# 8. Validate audio files
# =========================================================
def validate_audio_files(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collect basic info and detect problematic files.
    """
    validated_rows = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Validating audio"):
        file_path = row["filepath"]
        validated_row = row.to_dict()

        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)

            if y is None or len(y) == 0:
                raise ValueError("Empty audio signal.")

            duration = len(y) / sr if sr and sr > 0 else 0.0
            clipping_ratio = float(np.mean(np.abs(y) >= 0.999)) if len(y) > 0 else 0.0
            nan_present = bool(np.isnan(y).any())
            silence_ratio = float(np.mean(np.abs(y) < 1e-4)) if len(y) > 0 else 0.0

            validated_row["sample_rate_original"] = sr
            validated_row["duration_sec"] = duration
            validated_row["num_samples"] = len(y)
            validated_row["clipping_ratio"] = clipping_ratio
            validated_row["silence_ratio"] = silence_ratio
            validated_row["has_nan"] = nan_present
            validated_row["valid"] = not nan_present

            if nan_present:
                validated_row["error"] = "NaN values found in audio."

        except Exception as e:
            validated_row["sample_rate_original"] = None
            validated_row["duration_sec"] = None
            validated_row["num_samples"] = None
            validated_row["clipping_ratio"] = None
            validated_row["silence_ratio"] = None
            validated_row["has_nan"] = None
            validated_row["valid"] = False
            validated_row["error"] = str(e)

        validated_rows.append(validated_row)

    return pd.DataFrame(validated_rows)


# =========================================================
# 9. Extract features from metadata
# =========================================================
def extract_features_from_metadata(
    metadata_df: pd.DataFrame,
    save_features: bool = False,
    feature_dir: Optional[str] = None,
    apply_augmentation: bool = False,
    augment_train_only: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract log-mel features from metadata rows.

    Returns:
        X: (N, n_mels, fixed_frames)
        kept_metadata: matching metadata rows
    """
    features = []
    kept_rows = []

    if save_features and feature_dir is not None:
        ensure_dir(feature_dir)

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting features"):
        file_path = row["filepath"]
        fname = row["filename"]
        split = row.get("split", "unknown")

        try:
            y = load_audio(file_path, sr=SAMPLE_RATE, trim=True, trim_top_db=30)

            use_aug = apply_augmentation and ((not augment_train_only) or (split == "train"))
            if use_aug:
                y = augment_audio(y, sr=SAMPLE_RATE)

            feat = extract_logmel(y)

            features.append(feat)
            kept_rows.append(row.to_dict())

            if save_features and feature_dir is not None:
                save_name = os.path.splitext(fname)[0] + ".npy"
                np.save(os.path.join(feature_dir, save_name), feat)

        except Exception as e:
            print(f"Feature extraction failed for {file_path}: {e}")

    if len(features) == 0:
        X = np.empty((0, N_MELS, FIXED_FRAMES), dtype=np.float32)
    else:
        X = np.array(features, dtype=np.float32)

    kept_metadata = pd.DataFrame(kept_rows)
    return X, kept_metadata


def duplicate_with_augmentation(
    metadata_df: pd.DataFrame,
    n_augmented_copies: int = 1
) -> pd.DataFrame:
    """
    Duplicate metadata rows to later create augmented feature copies.
    Useful for training only.
    """
    if metadata_df is None or metadata_df.empty or n_augmented_copies <= 0:
        return metadata_df.copy()

    copies = [metadata_df.copy()]
    for aug_idx in range(n_augmented_copies):
        temp = metadata_df.copy()
        temp["augmented_copy_id"] = aug_idx + 1
        copies.append(temp)

    out = pd.concat(copies, ignore_index=True)
    if "augmented_copy_id" not in out.columns:
        out["augmented_copy_id"] = 0

    out["augmented_copy_id"] = out["augmented_copy_id"].fillna(0).astype(int)
    return out


# =========================================================
# 10. Standardization
# =========================================================
def fit_feature_scaler(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit mean/std on training set only.
    """
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train is empty. Cannot fit feature scaler.")

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    return mean.astype(np.float32), std.astype(np.float32)


def transform_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply standardization using training mean/std.
    """
    if X is None or len(X) == 0:
        return X.astype(np.float32)

    X_norm = (X - mean) / (std + EPS)
    return X_norm.astype(np.float32)


# =========================================================
# 11. Add channel dimension for CNN
# =========================================================
def add_channel_dim(X: np.ndarray) -> np.ndarray:
    """
    (N, H, W) -> (N, 1, H, W)
    """
    if X is None or len(X) == 0:
        return np.empty((0, 1, N_MELS, FIXED_FRAMES), dtype=np.float32)

    return np.expand_dims(X, axis=1).astype(np.float32)


# =========================================================
# 12. Save outputs
# =========================================================
def save_preprocessed_data(
    output_dir: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_train_no_channel: np.ndarray,
    X_test_no_channel: np.ndarray,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray
) -> None:
    ensure_dir(output_dir)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "X_train_no_channel.npy"), X_train_no_channel)
    np.save(os.path.join(output_dir, "X_test_no_channel.npy"), X_test_no_channel)
    np.save(os.path.join(output_dir, "scaler_mean.npy"), mean)
    np.save(os.path.join(output_dir, "scaler_std.npy"), std)

    train_metadata.to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
    test_metadata.to_csv(os.path.join(output_dir, "test_metadata.csv"), index=False)


# =========================================================
# 13. Full pipeline
# =========================================================
def run_preprocessing_pipeline(
    data_root: str,
    machine_type: str = "fan",
    output_dir: str = "../data/processed/fan",
    metadata_dir: str = "../data/metadata",
    train_normals_only: bool = True,
    save_individual_features: bool = False,
    apply_augmentation: bool = True,
    n_augmented_copies: int = 1,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline:
    - build metadata
    - validate files
    - split train/test
    - optionally keep only normal train files
    - optional train augmentation to simulate domain shift
    - extract log-mel features
    - fit scaler on train only
    - normalize train/test
    - save both raw normalized and CNN-ready versions
    """
    set_seed(seed)
    ensure_dir(output_dir)
    ensure_dir(metadata_dir)

    # 1) Metadata
    metadata_df = build_metadata(data_root=data_root, machine_type=machine_type)
    metadata_path = os.path.join(metadata_dir, f"metadata_{machine_type}.csv")
    metadata_df.to_csv(metadata_path, index=False)

    # 2) Validation
    metadata_validated = validate_audio_files(metadata_df)
    validated_path = os.path.join(metadata_dir, f"metadata_{machine_type}_validated.csv")
    metadata_validated.to_csv(validated_path, index=False)

    metadata_validated = metadata_validated[metadata_validated["valid"]].copy()

    if metadata_validated.empty:
        raise ValueError("No valid audio files found after validation.")

    unknown_count = int((metadata_validated["label"] == "unknown").sum())
    if unknown_count > 0:
        print(f"Warning: {unknown_count} files have unknown labels.")

    # 3) Split metadata
    if train_normals_only:
        train_metadata = metadata_validated[
            (metadata_validated["split"] == "train") &
            (metadata_validated["label"] == "normal")
        ].copy()
    else:
        train_metadata = metadata_validated[
            metadata_validated["split"] == "train"
        ].copy()

    test_metadata = metadata_validated[
        metadata_validated["split"] == "test"
    ].copy()

    print(f"Train files before augmentation: {len(train_metadata)}")
    print(f"Test files: {len(test_metadata)}")

    if train_metadata.empty:
        raise ValueError("Training metadata is empty. Check train split and label parsing.")
    if test_metadata.empty:
        raise ValueError("Test metadata is empty. Check test split and dataset structure.")

    # 4) Optional augmentation duplication for train only
    if apply_augmentation and n_augmented_copies > 0:
        train_metadata_for_extraction = duplicate_with_augmentation(
            train_metadata,
            n_augmented_copies=n_augmented_copies
        )
    else:
        train_metadata_for_extraction = train_metadata.copy()
        if "augmented_copy_id" not in train_metadata_for_extraction.columns:
            train_metadata_for_extraction["augmented_copy_id"] = 0

    if "augmented_copy_id" not in test_metadata.columns:
        test_metadata["augmented_copy_id"] = 0

    print(f"Train files after augmentation duplication: {len(train_metadata_for_extraction)}")

    # 5) Feature extraction
    train_feature_dir = os.path.join(output_dir, "individual_train_features") if save_individual_features else None
    test_feature_dir = os.path.join(output_dir, "individual_test_features") if save_individual_features else None

    X_train_raw, train_metadata_kept = extract_features_from_metadata(
        train_metadata_for_extraction,
        save_features=save_individual_features,
        feature_dir=train_feature_dir,
        apply_augmentation=apply_augmentation,
        augment_train_only=True
    )

    X_test_raw, test_metadata_kept = extract_features_from_metadata(
        test_metadata,
        save_features=save_individual_features,
        feature_dir=test_feature_dir,
        apply_augmentation=False,
        augment_train_only=True
    )

    print("Raw feature shapes:")
    print("X_train_raw:", X_train_raw.shape)
    print("X_test_raw :", X_test_raw.shape)

    if len(X_train_raw) == 0:
        raise ValueError("No training features extracted. Check preprocessing and dataset files.")
    if len(X_test_raw) == 0:
        raise ValueError("No test features extracted. Check preprocessing and dataset files.")

    # 6) Fit scaler on train only
    mean, std = fit_feature_scaler(X_train_raw)

    # 7) Normalize
    X_train_norm = transform_features(X_train_raw, mean, std)
    X_test_norm = transform_features(X_test_raw, mean, std)

    # Keep versions without channel for LOF / classical ML
    X_train_no_channel = X_train_norm.copy()
    X_test_no_channel = X_test_norm.copy()

    # 8) Add channel dim for CNN
    X_train = add_channel_dim(X_train_norm)
    X_test = add_channel_dim(X_test_norm)

    print("Final shapes after normalization + channel:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # 9) Save
    save_preprocessed_data(
        output_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
        X_train_no_channel=X_train_no_channel,
        X_test_no_channel=X_test_no_channel,
        train_metadata=train_metadata_kept,
        test_metadata=test_metadata_kept,
        mean=mean,
        std=std
    )

    print(f"Preprocessed data saved to: {output_dir}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_no_channel": X_train_no_channel,
        "X_test_no_channel": X_test_no_channel,
        "train_metadata": train_metadata_kept,
        "test_metadata": test_metadata_kept,
        "mean": mean,
        "std": std
    }


# =========================================================
# 14. Main
# =========================================================
if __name__ == "__main__":
    results = run_preprocessing_pipeline(
        data_root="../data",
        machine_type="fan",
        output_dir="../data/processed/fan",
        metadata_dir="../data/metadata",
        train_normals_only=True,
        save_individual_features=False,
        apply_augmentation=True,
        n_augmented_copies=1,
        seed=42
    )

    print("Preprocessing complete.")
