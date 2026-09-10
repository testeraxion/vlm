from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from vision_language_localization.config import PipelineConfig
from vision_language_localization.data.kitti_loader import iter_sequences
from vision_language_localization.evaluation.explanation_scoring import score_explanation_against_evidence
from vision_language_localization.evaluation.vlm_evaluation import compute_consistency, compute_latency_stats
from vision_language_localization.evaluation.metrics import compute_trajectory_metrics
from vision_language_localization.runtime_verification.stl_monitor import evaluate_runtime_properties
from vision_language_localization.sensor_fusion.ekf_fusion import (
    EkfFusion,
    EkfFusionConfig,
    LightweightFusion,
    compute_aligned_slam,
    compute_gps_only,
)
from vision_language_localization.slam.mock_slam import MockVisualSlam
from vision_language_localization.slam.trajectory_import import load_external_slam_output
from vision_language_localization.vlm.hf_vlm import HfLocalVLM, HfLocalVLMConfig
from vision_language_localization.vlm.qwen_vl import QwenVLMConfig, QwenVLVLM
from vision_language_localization.vlm.rule_based_vlm import RuleBasedVLM


class VLMBackend(Protocol):
    def explain(
        self,
        ate_rmse: float,
        drift_final_m: float,
        mean_reprojection_error: float,
        mean_feature_count: float,
        violation_count: int,
        scene_image_path: Path | None = None,
    ):
        ...


def _resolve_fusion(config: PipelineConfig):
    if config.fusion_method == "ekf":
        return EkfFusion(
            EkfFusionConfig(
                gps_noise_m=config.fusion_gps_noise_m,
                slam_noise_m=config.fusion_slam_noise_m,
                dead_reckoning_noise_mps=config.fusion_dead_reckoning_noise_mps,
                process_noise_accel_mps2=config.fusion_process_noise_accel_mps2,
                gate_chi2=config.fusion_gate_chi2,
                align_slam_to_gps=config.fusion_align_slam_to_gps,
                registration_window=config.fusion_registration_window,
            )
        )
    if config.fusion_method == "blend":
        return LightweightFusion(alpha_visual=config.fusion_alpha)
    raise ValueError(f"Unsupported fusion_method: {config.fusion_method}")


def _resolve_vlm_backend(config: PipelineConfig) -> VLMBackend:
    if config.vlm_backend == "rule_based":
        return RuleBasedVLM(consistency_runs=config.vlm_consistency_runs)

    if config.vlm_backend == "hf_local":
        return HfLocalVLM(HfLocalVLMConfig(model_name=config.vlm_model_name))

    if config.vlm_backend == "qwen_vl":
        model_name = config.vlm_model_name
        if model_name == "Salesforce/blip-image-captioning-base":
            model_name = config.qwen_default_model_name
        return QwenVLVLM(
            QwenVLMConfig(
                model_name=model_name,
                max_new_tokens=config.vlm_max_new_tokens,
                temperature=config.vlm_temperature,
            )
        )

    raise ValueError(f"Unsupported vlm_backend: {config.vlm_backend}")


def _run_vlm(vlm: VLMBackend, runs: int, **kwargs):
    """Run the VLM backend `runs` times, returning the first result, all
    explanations, and per-call latencies (used for consistency/latency eval)."""
    explanations: list[str] = []
    latencies: list[float] = []
    first_result = None
    for i in range(max(1, runs)):
        start = time.perf_counter()
        result = vlm.explain(**kwargs)
        latencies.append(time.perf_counter() - start)
        explanations.append(result.explanation)
        if i == 0:
            first_result = result
    return first_result, explanations, latencies


def run_pipeline(config: PipelineConfig) -> Path:
    config.output_root.mkdir(parents=True, exist_ok=True)

    mock_slam = MockVisualSlam(seed=config.random_seed)
    fusion = _resolve_fusion(config)
    vlm = _resolve_vlm_backend(config)

    sequences = list(iter_sequences(config.dataset_root, max_frames=config.max_frames_per_sequence))
    if config.sequence_id is not None:
        sequences = [s for s in sequences if s.sequence_id == config.sequence_id]
        if not sequences:
            raise ValueError(f"Sequence not found: {config.sequence_id}")
    if config.max_sequences is not None:
        sequences = sequences[: config.max_sequences]

    reports = []
    total_sequences = len(sequences)
    for seq_index, seq in enumerate(sequences, start=1):
        print(f"[{seq_index}/{total_sequences}] Processing sequence {seq.sequence_id}", flush=True)

        if config.slam_backend == "mock":
            slam_out = mock_slam.run(seq)
        elif config.slam_backend == "trajectory_file":
            trajectory_file = config.trajectory_root / f"{seq.sequence_id}{config.trajectory_file_suffix}"
            if trajectory_file.exists():
                slam_out = load_external_slam_output(trajectory_file=trajectory_file, gt_xyz=seq.gt_xyz)
            elif config.strict_trajectory_matching:
                raise FileNotFoundError(f"Missing trajectory file for {seq.sequence_id}: {trajectory_file}")
            else:
                slam_out = mock_slam.run(seq)
        else:
            raise ValueError(f"Unsupported slam_backend: {config.slam_backend}")

        n = min(len(seq.frames), len(slam_out.est_xyz))
        gt_xyz = seq.gt_xyz[:n]
        slam_xyz = slam_out.est_xyz[:n]
        tracking_ok = slam_out.tracking_ok[:n]
        reproj = slam_out.reprojection_error[:n]
        features = slam_out.feature_count[:n]

        fusion_out = fusion.run(seq, slam_out)
        fused_xyz = fusion_out.fused_xyz[:n]

        slam_metrics = compute_trajectory_metrics(gt_xyz, slam_xyz)
        fusion_metrics = compute_trajectory_metrics(gt_xyz, fused_xyz)

        # Ablation baselines
        aligned_slam_out = compute_aligned_slam(seq, slam_out, fusion.config if hasattr(fusion, 'config') else None)
        aligned_slam_xyz = aligned_slam_out.fused_xyz[:n]
        aligned_slam_metrics = compute_trajectory_metrics(gt_xyz, aligned_slam_xyz)

        gps_only_out = compute_gps_only(seq)
        gps_only_xyz = gps_only_out.fused_xyz[:n]
        gps_only_metrics = compute_trajectory_metrics(gt_xyz, gps_only_xyz)

        pos_error = np.linalg.norm(fused_xyz - gt_xyz, axis=1)
        scene_idx = int(np.argmax(pos_error))
        scene_image_path = None
        if scene_idx < len(seq.frames):
            scene_image_path = seq.frames[scene_idx].image_path

        dt_s = 0.1
        rv = evaluate_runtime_properties(
            position_error_m=pos_error,
            tracking_ok=tracking_ok,
            dt_s=dt_s,
            max_position_error_m=config.max_position_error_m,
            max_lost_duration_s=config.max_lost_duration_s,
            recovery_window_s=config.recovery_window_s,
        )

        print(
            f"[{seq_index}/{total_sequences}] Running {config.vlm_backend} inference "
            f"({config.vlm_consistency_runs}x) for {seq.sequence_id}",
            flush=True,
        )
        vlm_result, explanations, latencies = _run_vlm(
            vlm,
            config.vlm_consistency_runs,
            ate_rmse=slam_metrics.ate_rmse,
            drift_final_m=slam_metrics.drift_final_m,
            mean_reprojection_error=float(np.mean(reproj)),
            mean_feature_count=float(np.mean(features)),
            violation_count=rv.violation_count,
            scene_image_path=scene_image_path,
        )
        consistency = compute_consistency(explanations)
        latency = compute_latency_stats(latencies)

        evidence = score_explanation_against_evidence(
            explanation=vlm_result.explanation,
            mean_reprojection_error=float(np.mean(reproj)),
            mean_feature_count=float(np.mean(features)),
            drift_final_m=slam_metrics.drift_final_m,
            violation_count=rv.violation_count,
        )

        reports.append(
            {
                "sequence_id": seq.sequence_id,
                "num_frames": n,
                "backends": {
                    "slam_backend": config.slam_backend,
                    "vlm_backend": config.vlm_backend,
                },
                "scene_image_path": str(scene_image_path) if scene_image_path is not None else None,
                "slam_metrics": asdict(slam_metrics),
                "fusion_metrics": asdict(fusion_metrics),
                "aligned_slam_metrics": asdict(aligned_slam_metrics),
                "gps_only_metrics": asdict(gps_only_metrics),
                "runtime_verification": asdict(rv),
                "vlm": {
                    **asdict(vlm_result),
                    "consistency": consistency,
                    "latency": latency,
                },
                "explanation_evidence": asdict(evidence),
            }
        )
        print(f"[{seq_index}/{total_sequences}] Finished {seq.sequence_id}", flush=True)

    out_file = config.output_root / "latest_run.json"
    out_file.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return out_file
