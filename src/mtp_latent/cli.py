from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mtp_latent.config import ExperimentConfig
from mtp_latent.data import build_dataloaders, build_sft_dataloaders, build_transition_dataloaders
from mtp_latent.models import ReasoningCodec, ReasoningTransitionModel, SFTLanguageModel
from mtp_latent.training import evaluate_codec, train_codec, train_sft, train_transition
from mtp_latent.utils import build_tokenizer, cleanup_distributed, init_distributed, load_json, save_json, set_seed


def _load_codec_for_eval(codec: ReasoningCodec, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    codec.load_state_dict(state["model_state"])


def _init_transition_backbone_from_encoder_checkpoint(
    transition_model: ReasoningTransitionModel,
    checkpoint_path: str,
) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model_state"]
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder.* weights found in checkpoint: {checkpoint_path}")
    load_result = transition_model.backbone.load_state_dict(encoder_state, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(
            f"Unexpected encoder keys while initializing transition backbone from {checkpoint_path}: "
            f"{load_result.unexpected_keys}"
        )
    if load_result.missing_keys:
        print(
            "Warning: missing backbone keys during encoder initialization:",
            ", ".join(load_result.missing_keys),
        )


def run_train_codec(config_path: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    set_seed(config.train.seed)
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    rank = int(__import__("os").environ.get("RANK", "0"))
    _, loaders, tokenizer = build_dataloaders(config.data, config.model.tokenizer_name_or_path, world_size=world_size, rank=rank)
    config.model.vocab_size = len(tokenizer)
    codec = ReasoningCodec(config.model, pad_token_id=tokenizer.pad_token_id)
    best_path = train_codec(codec, loaders, config)
    print(best_path)


def run_train_transition(config_path: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    if not config.transition.codec_checkpoint:
        raise ValueError("transition.codec_checkpoint must be set for transition training.")
    if not config.model.model_name_or_path:
        raise ValueError("transition training requires model.model_name_or_path to initialize the GPT-2 transition backbone.")
    set_seed(config.train.seed)
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    rank = int(__import__("os").environ.get("RANK", "0"))

    dist_ctx = init_distributed(config.train.device, config.train.distributed_backend)
    try:
        codec_tokenizer_name = config.model.tokenizer_name_or_path or config.model.model_name_or_path
        tokenizer = build_tokenizer(codec_tokenizer_name)
        config.model.vocab_size = len(tokenizer)
        codec = ReasoningCodec(config.model, pad_token_id=tokenizer.pad_token_id)
        _load_codec_for_eval(codec, config.transition.codec_checkpoint)
        codec.to(dist_ctx.device)
        codec.eval()
        _, transition_loaders, transition_tokenizer = build_transition_dataloaders(
            config.data,
            codec_tokenizer_name,
            world_size=world_size,
            rank=rank,
        )
        config.model.vocab_size = len(transition_tokenizer)
        transition_model = ReasoningTransitionModel(config.model, config.transition, vocab_size=len(transition_tokenizer))
        if config.transition.encoder_init_checkpoint:
            _init_transition_backbone_from_encoder_checkpoint(
                transition_model,
                config.transition.encoder_init_checkpoint,
            )
        if config.transition.init_checkpoint:
            state = torch.load(config.transition.init_checkpoint, map_location="cpu")
            transition_model.load_state_dict(state["model_state"])
        best_path = train_transition(transition_model, codec, transition_loaders, config)
        print(best_path)
    finally:
        cleanup_distributed()


def run_train_sft(config_path: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    set_seed(config.train.seed)
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    rank = int(__import__("os").environ.get("RANK", "0"))
    _, loaders, tokenizer = build_sft_dataloaders(
        config.data,
        config.model.tokenizer_name_or_path,
        config.sft.task,
        world_size=world_size,
        rank=rank,
    )
    config.model.vocab_size = len(tokenizer)
    model = SFTLanguageModel(config.model)
    best_path = train_sft(model, loaders, config)
    print(best_path)


def run_evaluate(config_path: str, codec_checkpoint: str) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    set_seed(config.train.seed)
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    rank = int(__import__("os").environ.get("RANK", "0"))
    _, loaders, tokenizer = build_dataloaders(config.data, config.model.tokenizer_name_or_path, world_size=world_size, rank=rank)
    config.model.vocab_size = len(tokenizer)
    codec = ReasoningCodec(config.model, pad_token_id=tokenizer.pad_token_id)
    _load_codec_for_eval(codec, codec_checkpoint)
    dist_ctx = init_distributed(config.train.device, config.train.distributed_backend)
    device = dist_ctx.device
    codec.to(device)

    try:
        codec_metrics = evaluate_codec(codec, loaders["test"], config, device)
        results = {
            "codec_test_loss": codec_metrics.loss,
            "codec_metrics": codec_metrics.metrics,
        }

        if dist_ctx.is_main_process:
            output_dir = Path(config.train.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(output_dir / "evaluation.json", results)
            print(output_dir / "evaluation.json")
    finally:
        cleanup_distributed()


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
        "first_future_kinds": sample.future_kinds,
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

    sft_parser = subparsers.add_parser("train-sft")
    sft_parser.add_argument("--config", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--codec-checkpoint", required=True)

    history_parser = subparsers.add_parser("show-history")
    history_parser.add_argument("--path", required=True)

    inspect_parser = subparsers.add_parser("inspect-data")
    inspect_parser.add_argument("--config", required=True)

    args = parser.parse_args()

    if args.command == "train-codec":
        run_train_codec(args.config)
    elif args.command == "train-transition":
        run_train_transition(args.config)
    elif args.command == "train-sft":
        run_train_sft(args.config)
    elif args.command == "evaluate":
        run_evaluate(args.config, args.codec_checkpoint)
    elif args.command == "show-history":
        run_show_history(args.path)
    elif args.command == "inspect-data":
        run_inspect_data(args.config)


if __name__ == "__main__":
    main()
