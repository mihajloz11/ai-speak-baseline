from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import FeatureConfig
from .model import ModelConfig
from .pipeline import TrainConfig, infer_directory, prepare_data, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Speak baseline pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data", help="Unpack archives and cache features")
    prepare_parser.add_argument("--raw-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--extract-avatar", action="store_true")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--sample-rate", type=int, default=16000)
    prepare_parser.add_argument("--fps", type=int, default=60)
    prepare_parser.add_argument("--n-mels", type=int, default=80)

    train_parser = subparsers.add_parser("train", help="Train the baseline model")
    train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--artifact-root", type=Path, required=True)
    train_parser.add_argument("--sample-rate", type=int, default=16000)
    train_parser.add_argument("--fps", type=int, default=60)
    train_parser.add_argument("--n-mels", type=int, default=80)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=6)
    train_parser.add_argument("--dropout", type=float, default=0.15)
    train_parser.add_argument("--device", type=str, default=None)

    infer_parser = subparsers.add_parser("infer", help="Run inference and export CSV/meta.json")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--input-dir", type=Path, required=True)
    infer_parser.add_argument("--output-dir", type=Path, required=True)
    infer_parser.add_argument("--fps-out", type=int, default=60)
    infer_parser.add_argument("--device", type=str, default=None)
    infer_parser.add_argument("--default-speaker", type=str, default="spk08")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-data":
        feature_config = FeatureConfig(sample_rate=args.sample_rate, fps=args.fps, n_mels=args.n_mels)
        result = prepare_data(
            raw_root=args.raw_root,
            output_root=args.output_root,
            feature_config=feature_config,
            extract_avatar=args.extract_avatar,
            force=args.force,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "train":
        feature_config = FeatureConfig(sample_rate=args.sample_rate, fps=args.fps, n_mels=args.n_mels)
        model_config = ModelConfig(input_dim=0, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
        train_config = TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        result = train_model(
            data_root=args.data_root,
            artifact_root=args.artifact_root,
            feature_config=feature_config,
            model_config=model_config,
            train_config=train_config,
            device=args.device,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "infer":
        result = infer_directory(
            checkpoint_path=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            fps_out=args.fps_out,
            device=args.device,
            default_speaker=args.default_speaker,
        )
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
