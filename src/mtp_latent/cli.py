from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mtp_latent.config import ExperimentConfig
from mtp_latent.data import build_dataloaders
from mtp_latent.models import LatentTransitionModel, ReasoningCodec
from mtp_latent.training import evaluate_codec, evaluate_transition, train_codec, train_transition
from mtp_latent.utils import load_json, save_json, set_seed


def _load_codec_for_eval(codec: ReasoningCodec, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    codec.load_state_dict(state["model_state"])


def _load_transition_for_eval(transition: LatentTransitionModel, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    transition.load_state_dict(state["transition_state"])


def run_train_codec(config_path: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    set_seed(config.train.seed)
    _, loaders, tokenizer = build_dataloaders(config.data, config.model.tokenizer_name_or_path)
    config.model.vocab_size = len(tokenizer)
    codec = ReasoningCodec(config.model, pad_token_id=tokenizer.pad_token_id)
    best_path = train_codec(codec, loaders, config)
    print(best_path)


def run_train_transition(config_path: str, codec_checkpoint: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    set_seed(config.train.seed)
    _, loaders, tokenizer = build_dataloaders(config.data, config.model.tokenizer_name_or_path)
    config.model.vocab_size = len(tokenizer)
    codec = ReasoningCodec(config.model, pad_token_id=tokenizer.pad_token_id)
    _load_codec_for_eval(codec, codec_checkpoint)
    transition = LatentTransitionModel(config.model.latent_dim, config.transition)
    best_path = train_transition(codec, transition, loaders, config)
    print(best_path)


def run_evaluate(config_path: str, codec_checkpoint: str, transition_checkpoint: str | None) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    set_seed(config.train.seed)
    _, loaders, tokenizer = build_dataloaders(config.data, config.model.tokenizer_name_or_path)
    config.model.vocab_size = len(tokenizer)
    codec = ReasoningCodec(config.model, pad_token_id=tokenizer.pad_token_id)
    _load_codec_for_eval(codec, codec_checkpoint)
    device = torch.device(config.train.device)
    codec.to(device)

    codec_metrics = evaluate_codec(codec, loaders["test"], config, device)
    results = {
        "codec_test_loss": codec_metrics.loss,
        "codec_metrics": codec_metrics.metrics,
    }

    if transition_checkpoint:
        transition = LatentTransitionModel(config.model.latent_dim, config.transition)
        _load_transition_for_eval(transition, transition_checkpoint)
        transition.to(device)
        from mtp_latent.training import build_transition_pairs

        source_test, target_test = build_transition_pairs(codec, loaders["test"], device)
        transition_metrics = evaluate_transition(transition, source_test, target_test, device)
        results["transition_test_loss"] = transition_metrics.loss
        results["transition_metrics"] = transition_metrics.metrics

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "evaluation.json", results)
    print(output_dir / "evaluation.json")


def run_show_history(path: str) -> None:
    data = load_json(path)
    print(data)


def run_inspect_data(config_path: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    dataset, loaders, tokenizer = build_dataloaders(config.data, config.model.tokenizer_name_or_path)
    sample = dataset[0]
    preview = {
        "num_records": len(dataset.records),
        "num_samples": len(dataset),
        "tokenizer_vocab_size": len(tokenizer),
        "first_prefix": sample.prefix_text,
        "first_future_steps": sample.future_steps,
        "first_answer": sample.answer,
        "train_batches": len(loaders["train"]),
        "valid_batches": len(loaders["valid"]),
        "test_batches": len(loaders["test"]),
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="MTP latent reasoning experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    codec_parser = subparsers.add_parser("train-codec")
    codec_parser.add_argument("--config", required=True)

    transition_parser = subparsers.add_parser("train-transition")
    transition_parser.add_argument("--config", required=True)
    transition_parser.add_argument("--codec-checkpoint", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--codec-checkpoint", required=True)
    evaluate_parser.add_argument("--transition-checkpoint")

    history_parser = subparsers.add_parser("show-history")
    history_parser.add_argument("--path", required=True)

    inspect_parser = subparsers.add_parser("inspect-data")
    inspect_parser.add_argument("--config", required=True)

    args = parser.parse_args()

    if args.command == "train-codec":
        run_train_codec(args.config)
    elif args.command == "train-transition":
        run_train_transition(args.config, args.codec_checkpoint)
    elif args.command == "evaluate":
        run_evaluate(args.config, args.codec_checkpoint, args.transition_checkpoint)
    elif args.command == "show-history":
        run_show_history(args.path)
    elif args.command == "inspect-data":
        run_inspect_data(args.config)


if __name__ == "__main__":
    main()
