from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision_language_localization.config import PipelineConfig
from vision_language_localization.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vision-Language localization research pipeline")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Path to KITTI-style dataset root")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Directory for reports")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames per sequence")
    parser.add_argument("--max-sequences", type=int, default=None, help="Max number of sequences to process")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic simulation")
    parser.add_argument(
        "--slam-backend",
        type=str,
        default="mock",
        choices=["mock", "trajectory_file"],
        help="SLAM backend: mock simulator or external trajectory-file ingestion",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path("slam"),
        help="Folder containing trajectory files named <sequence_id>.kitti.txt",
    )
    parser.add_argument(
        "--trajectory-file-suffix",
        type=str,
        default=".kitti.txt",
        help="Suffix used to resolve trajectory files",
    )
    parser.add_argument(
        "--strict-trajectory-matching",
        action="store_true",
        help="Fail if a sequence trajectory file is missing when using trajectory_file backend",
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        default=None,
        help="Process only this specific sequence (e.g. 2011_09_26_drive_0009_sync)",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="ekf",
        choices=["ekf", "blend"],
        help="Sensor fusion: EKF (SLAM + GPS + IMU dead-reckoning) or legacy weighted blend",
    )
    parser.add_argument(
        "--vlm-backend",
        type=str,
        default="rule_based",
        choices=["rule_based", "hf_local", "qwen_vl"],
        help="VLM backend: deterministic baseline, local image-captioning model, or Qwen multimodal model",
    )
    parser.add_argument(
        "--vlm-model-name",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="Model name for selected local VLM backend",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=180,
        help="Max new tokens for generative VLM backends",
    )
    parser.add_argument(
        "--vlm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generative VLM backends",
    )
    parser.add_argument(
        "--vlm-consistency-runs",
        type=int,
        default=1,
        help="Repeated runs of the VLM on the same scene to measure consistency (1 disables)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        max_frames_per_sequence=args.max_frames,
        max_sequences=args.max_sequences,
        random_seed=args.seed,
        slam_backend=args.slam_backend,
        trajectory_root=args.trajectory_root,
        trajectory_file_suffix=args.trajectory_file_suffix,
        strict_trajectory_matching=args.strict_trajectory_matching,
        fusion_method=args.fusion_method,
        vlm_backend=args.vlm_backend,
        vlm_model_name=args.vlm_model_name,
        vlm_max_new_tokens=args.vlm_max_new_tokens,
        vlm_temperature=args.vlm_temperature,
        vlm_consistency_runs=args.vlm_consistency_runs,
        sequence_id=args.sequence_id,
    )
    report_path = run_pipeline(cfg)
    print(f"Pipeline complete. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
