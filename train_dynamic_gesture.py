import argparse
import json
from pathlib import Path

from gesture_runtime.temporal_model import TemporalTrainConfig, train_temporal_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dynamic gesture temporal model")
    parser.add_argument("--train-manifest", type=str, required=True, help="JSON/JSONL train manifest")
    parser.add_argument("--val-manifest", type=str, required=True, help="JSON/JSONL validation manifest")
    parser.add_argument("--output-dir", type=str, default="gesture_checkpoints", help="checkpoint output directory")
    parser.add_argument("--window-size", type=int, default=32, help="sequence window length")
    parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
    parser.add_argument("--epochs", type=int, default=40, help="number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="optimizer weight decay")
    parser.add_argument("--hidden-dim", type=int, default=192, help="transformer hidden dim")
    parser.add_argument("--num-layers", type=int, default=3, help="transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="transformer heads")
    parser.add_argument("--dropout", type=float, default=0.15, help="dropout ratio")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="training device")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TemporalTrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    output_path = train_temporal_model(config)
    meta_path = Path(args.output_dir) / "train_config.json"
    meta_path.write_text(json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best checkpoint:", str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
