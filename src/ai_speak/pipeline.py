from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .constants import BLENDSHAPE_DIM, DEFAULT_FPS
from .data import (
    FeatureConfig,
    build_manifest,
    build_text_vocab,
    encode_text,
    ensure_directory,
    extract_frame_features,
    precompute_features,
    read_blendshape_csv,
    safe_unzip,
    save_dataset_info,
    save_json,
    split_manifest,
)
from .model import BlendshapeRegressor, ModelConfig, build_channel_weights, masked_huber_loss


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    validation_ratio: float = 0.15
    velocity_loss_weight: float = 0.4
    gradient_clip: float = 1.0
    early_stopping_patience: int = 8
    seed: int = 42
    num_workers: int = 2


class BlendshapeDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        text_vocab: dict[str, int],
        speaker_to_index: dict[str, int],
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.text_vocab = text_vocab
        self.speaker_to_index = speaker_to_index
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        features = np.load(row.feature_path).astype(np.float32)
        target = read_blendshape_csv(Path(row.target_path)).astype(np.float32)
        target = (target - self.target_mean) / self.target_std
        transcript = "" if pd.isna(row.transcript) else str(row.transcript)
        text_tokens = encode_text(transcript, self.text_vocab)
        return {
            "sample_id": row.sample_id,
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "speaker_id": torch.tensor(self.speaker_to_index[row.speaker], dtype=torch.long),
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
        }


def _collate_batch(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[str]]:
    batch_size = len(batch)
    frame_lengths = [item["features"].shape[0] for item in batch]
    max_frames = max(frame_lengths)
    feature_dim = batch[0]["features"].shape[1]
    target_dim = batch[0]["target"].shape[1]

    features = torch.zeros(batch_size, max_frames, feature_dim, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_frames, target_dim, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)

    text_lengths = [int(item["text_tokens"].numel()) for item in batch]
    max_text = max(max(text_lengths), 1)
    text_tokens = torch.zeros(batch_size, max_text, dtype=torch.long)
    text_mask = torch.zeros(batch_size, max_text, dtype=torch.bool)

    speaker_ids = torch.stack([item["speaker_id"] for item in batch], dim=0)
    sample_ids = [str(item["sample_id"]) for item in batch]

    for batch_index, item in enumerate(batch):
        length = item["features"].shape[0]
        features[batch_index, :length] = item["features"]
        targets[batch_index, :length] = item["target"]
        frame_mask[batch_index, :length] = True

        if item["text_tokens"].numel() > 0:
            text_length = item["text_tokens"].numel()
            text_tokens[batch_index, :text_length] = item["text_tokens"]
            text_mask[batch_index, :text_length] = True

    return {
        "sample_ids": sample_ids,
        "features": features,
        "targets": targets,
        "frame_mask": frame_mask,
        "speaker_ids": speaker_ids,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_stats(train_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(BLENDSHAPE_DIM, dtype=np.float64)
    total_sq = np.zeros(BLENDSHAPE_DIM, dtype=np.float64)
    count = 0
    for row in train_rows.itertuples(index=False):
        target = read_blendshape_csv(Path(row.target_path)).astype(np.float64)
        total += target.sum(axis=0)
        total_sq += np.square(target).sum(axis=0)
        count += target.shape[0]

    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - np.square(mean), 1e-6)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def _masked_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[1] < 2:
        return prediction.new_tensor(0.0)
    velocity_prediction = prediction[:, 1:] - prediction[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    velocity_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
    return masked_huber_loss(velocity_prediction, velocity_target, velocity_mask, channel_weights)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    velocity_loss_weight: float,
    gradient_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    channel_weights = build_channel_weights(device)
    losses: list[float] = []

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        speaker_ids = batch["speaker_ids"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        text_mask = batch["text_mask"].to(device)

        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            prediction = model(features, speaker_ids, text_tokens=text_tokens, text_mask=text_mask)
            reconstruction = masked_huber_loss(prediction, targets, frame_mask, channel_weights)
            velocity = _masked_velocity_loss(prediction, targets, frame_mask, channel_weights)
            loss = reconstruction + velocity_loss_weight * velocity

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def prepare_data(
    raw_root: str | Path,
    output_root: str | Path,
    feature_config: FeatureConfig,
    extract_avatar: bool = False,
    force: bool = False,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, object]:
    raw_root = Path(raw_root)
    output_root = ensure_directory(Path(output_root))
    extracted_root = ensure_directory(output_root / "extracted")
    features_root = ensure_directory(output_root / "features")
    splits_root = ensure_directory(output_root / "splits")

    required_archives = {
        "spk08_blendshapes.zip": extracted_root / "spk08_blendshapes",
        "spk14_blendshapes.zip": extracted_root / "spk14_blendshapes",
        "labels_aligned.zip": extracted_root / "labels_aligned",
    }
    if extract_avatar:
        required_archives["avatar.zip"] = extracted_root / "avatar"

    for archive_name, destination in required_archives.items():
        archive_path = raw_root / archive_name
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing required archive: {archive_path}")
        safe_unzip(archive_path, destination, force=force)

    manifest = build_manifest(extracted_root)
    manifest = precompute_features(manifest, features_root, feature_config, force=force)
    train_df, val_df = split_manifest(manifest, validation_ratio=validation_ratio, seed=seed)
    vocab = build_text_vocab(train_df["transcript"].fillna("").astype(str).tolist())

    manifest_path = output_root / "manifest.csv"
    train_path = splits_root / "train.csv"
    val_path = splits_root / "val.csv"
    manifest.to_csv(manifest_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    example_features = np.load(manifest.iloc[0]["feature_path"])
    save_dataset_info(output_root / "dataset_info.json", manifest, feature_config, example_features.shape[1], vocab)

    return {
        "output_root": output_root.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "train_path": train_path.as_posix(),
        "val_path": val_path.as_posix(),
        "feature_dim": int(example_features.shape[1]),
        "speakers": sorted(manifest["speaker"].unique().tolist()),
        "vocab_size": len(vocab),
    }


def train_model(
    data_root: str | Path,
    artifact_root: str | Path,
    feature_config: FeatureConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: str | None = None,
) -> dict[str, object]:
    _seed_everything(train_config.seed)
    data_root = Path(data_root)
    artifact_root = ensure_directory(Path(artifact_root))
    checkpoints_root = ensure_directory(artifact_root / "checkpoints")
    logs_root = ensure_directory(artifact_root / "logs")

    train_rows = pd.read_csv(data_root / "splits" / "train.csv")
    val_rows = pd.read_csv(data_root / "splits" / "val.csv")
    dataset_info = json.loads((data_root / "dataset_info.json").read_text(encoding="utf-8"))

    speakers = dataset_info["speakers"]
    text_vocab = dataset_info["text_vocab"]
    speaker_to_index = {speaker: index for index, speaker in enumerate(speakers)}

    target_mean, target_std = _target_stats(train_rows)
    model_config.input_dim = int(dataset_info["feature_dim"])
    model_config.speaker_count = len(speakers)
    model_config.vocab_size = len(text_vocab)

    train_dataset = BlendshapeDataset(train_rows, text_vocab, speaker_to_index, target_mean, target_std)
    val_dataset = BlendshapeDataset(val_rows, text_vocab, speaker_to_index, target_mean, target_std)

    use_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=use_device.type == "cuda",
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=use_device.type == "cuda",
        collate_fn=_collate_batch,
    )

    model = BlendshapeRegressor(model_config).to(use_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_val_loss = float("inf")
    best_checkpoint_path = checkpoints_root / "best.pt"
    history: list[dict[str, float]] = []
    patience = 0

    for epoch in range(1, train_config.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=use_device,
            velocity_loss_weight=train_config.velocity_loss_weight,
            gradient_clip=train_config.gradient_clip,
        )
        val_loss = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=use_device,
            velocity_loss_weight=train_config.velocity_loss_weight,
            gradient_clip=train_config.gradient_clip,
        )

        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(epoch_record)
        save_json(logs_root / "history.json", {"history": history})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            checkpoint_payload = {
                "model_state": model.state_dict(),
                "model_config": asdict(model_config),
                "feature_config": asdict(feature_config),
                "train_config": asdict(train_config),
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
                "speakers": speakers,
                "speaker_to_index": speaker_to_index,
                "text_vocab": text_vocab,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
        else:
            patience += 1
            if patience >= train_config.early_stopping_patience:
                break

    return {
        "best_checkpoint": best_checkpoint_path.as_posix(),
        "best_val_loss": float(best_val_loss),
        "history_path": (logs_root / "history.json").as_posix(),
    }


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[BlendshapeRegressor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BlendshapeRegressor(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _smooth_predictions(prediction: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return prediction
    smoothed = prediction.copy()
    for frame_index in range(1, smoothed.shape[0]):
        smoothed[frame_index] = alpha * smoothed[frame_index - 1] + (1.0 - alpha) * smoothed[frame_index]
    return smoothed


def infer_directory(
    checkpoint_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    fps_out: int = DEFAULT_FPS,
    device: str | None = None,
    default_speaker: str = "spk08",
    smoothing_alpha: float = 0.15,
) -> dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = ensure_directory(Path(output_dir))
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = _load_checkpoint(Path(checkpoint_path), device_obj)

    feature_config = FeatureConfig(**checkpoint["feature_config"])
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    speaker_to_index = checkpoint["speaker_to_index"]
    text_vocab = checkpoint["text_vocab"]
    speaker_index = speaker_to_index.get(default_speaker, 0)

    results: list[dict[str, object]] = []

    for audio_path in sorted(input_dir.glob("*.wav")):
        transcript_path = audio_path.with_suffix(".txt")
        transcript = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else ""

        audio_duration = max(get_audio_duration(audio_path), 1e-6)
        model_frames = max(1, int(round(audio_duration * feature_config.fps)))
        start_time = time.perf_counter()
        features = extract_frame_features(audio_path, model_frames, feature_config)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device_obj)
        speaker_tensor = torch.tensor([speaker_index], dtype=torch.long, device=device_obj)
        encoded_text = encode_text(transcript, text_vocab)
        if not encoded_text:
            encoded_text = [0]
        text_tokens = torch.tensor([encoded_text], dtype=torch.long, device=device_obj)
        text_mask = text_tokens != 0

        with torch.no_grad():
            normalized = model(
                features_tensor,
                speaker_tensor,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )[0].detach().cpu().numpy()
        prediction = normalized * target_std + target_mean
        prediction = np.clip(prediction, 0.0, 1.0)
        prediction = _smooth_predictions(prediction, smoothing_alpha)
        inference_time = time.perf_counter() - start_time

        output_frames = max(1, int(round(audio_duration * fps_out)))
        if output_frames != prediction.shape[0]:
            source_x = np.linspace(0.0, 1.0, num=prediction.shape[0], dtype=np.float32)
            target_x = np.linspace(0.0, 1.0, num=output_frames, dtype=np.float32)
            prediction = np.stack(
                [np.interp(target_x, source_x, prediction[:, index]) for index in range(prediction.shape[1])],
                axis=1,
            ).astype(np.float32)

        csv_path = output_dir / f"{audio_path.stem}.csv"
        np.savetxt(csv_path, prediction, delimiter=",", fmt="%.6f")

        results.append(
            {
                "file": audio_path.name,
                "csv": csv_path.name,
                "inference_time_sec": round(float(inference_time), 6),
                "rtf": round(float(inference_time / audio_duration), 6),
                "lookahead_ms": 0,
                "fps_out": int(fps_out),
            }
        )

    meta = {
        "system": {
            "default_speaker": default_speaker,
            "lookahead_ms": 0,
            "fps_out": int(fps_out),
        },
        "files": results,
    }
    save_json(output_dir / "meta.json", meta)
    return meta


def get_audio_duration(audio_path: Path) -> float:
    import soundfile as sf

    info = sf.info(audio_path.as_posix())
    if info.samplerate:
        return float(info.frames / info.samplerate)
    return 0.0



