import enum
import logging
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.utils.bundled_inputs
import torch.utils.mobile_optimizer
import yaml

from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.utils import load_model


class ParallelWaveGANSimple(torch.nn.Module):
    """
    Simple wrapper Module for ParallelWaveGANGenerator
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model.simple_inference(feats)

    @torch.jit.export
    def inference(self, feats: torch.Tensor) -> torch.Tensor:
        return self.forward(feats)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export ParallelWaveGAN to TorchScript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Saved ParallelWaveGAN PyTorch model, e.g. parallelWaveGAN.pkl",
    )
    parser.add_argument(
        "model_config",
        type=Path,
        help="Generated training config for `model`, e.g. parallel_wavegan.v1.yaml",
    )
    parser.add_argument(
        "output_model",
        type=Path,
        help="Output path for exported model",
    )
    parser.add_argument(
        "--export-type",
        type=str,
        choices=("onnx", "lite", "full"),
        default="full",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    model = load_model(
        args.model, yaml.load(args.model_config.open(), Loader=yaml.SafeLoader)
    )
    model.remove_weight_norm()

    simple_model = ParallelWaveGANSimple(model)
    scripted_model = torch.jit.script(simple_model)
    scripted_model.eval()

    # TODO(rkjaran): Quantize

    if args.export_type == "lite":
        torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        scripted_model._save_for_lite_interpreter(str(args.output_model))
        logging.info(
            "Model exported for PyTorch Mobile Lite to '%s'", args.output_model
        )
    elif args.export_type == "full":
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        torch.jit.save(scripted_model, args.output_model)
        logging.info("Full optimized model exported to '%s'", args.output_model)
    elif args.export_type == "onnx":
        input_tensor = torch.randn([210, 80])
        torch.onnx.export(
            simple_model,
            input_tensor,
            args.output_model,
            input_names=["mel"],
            output_names=["wav"],
            dynamic_axes={
                "mel": [0],
                "wav": [0],
            },
            opset_version=11,
        )
        logging.info("Model exported to ONNX '%s'", args.output_model)


if __name__ == "__main__":
    main()
