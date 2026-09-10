from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    dataset_root: Path = Path("dataset")
    output_root: Path = Path("outputs")
    max_frames_per_sequence: int = 300
    max_sequences: int | None = None
    random_seed: int = 7

    # Runtime verification thresholds
    max_position_error_m: float = 1.0
    max_lost_duration_s: float = 2.0
    recovery_window_s: float = 5.0

    # Fusion controls
    fusion_method: str = "ekf"  # ekf (EKF: SLAM + GPS + IMU dead-reckoning) | blend (weighted baseline)
    fusion_alpha: float = 0.65  # blend-only: visual weight vs inertial prior
    fusion_gps_noise_m: float = 1.5
    fusion_slam_noise_m: float = 1.0
    fusion_dead_reckoning_noise_mps: float = 3.0
    fusion_process_noise_accel_mps2: float = 2.0
    fusion_gate_chi2: float = 0.0
    fusion_align_slam_to_gps: bool = True
    fusion_registration_window: int = 50

    # VLM behavior controls for baseline rule-based model.
    # Consistency is measured by running the backend N times on the same scene;
    # keep at 1 for fast runs and raise it (e.g. 5) for consistency evaluation.
    vlm_consistency_runs: int = 1

    # Backend configuration
    slam_backend: str = "mock"  # mock | trajectory_file
    vlm_backend: str = "rule_based"  # rule_based | hf_local | qwen_vl

    # External trajectory ingestion
    trajectory_root: Path = Path("slam")
    trajectory_file_suffix: str = ".kitti.txt"
    strict_trajectory_matching: bool = False

    # Optional local VLM adapter
    vlm_model_name: str = "Salesforce/blip-image-captioning-base"
    qwen_default_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_max_new_tokens: int = 180
    vlm_temperature: float = 0.0

    # Sequence filtering
    sequence_id: str | None = None  # Process only this sequence
