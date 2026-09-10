"""Run Qwen2.5-VL modality ablation on failure events.

Tests four input conditions:
1. image_only: VLM receives ONLY the scene image (metrics zeroed)
2. metrics_only: VLM receives ONLY numeric metrics (no image)
3. combined: VLM receives both image and metrics (baseline)
4. contradictory: VLM receives image but metrics are intentionally inflated

For each condition, we measure:
- Explanation text, detected hazards, and scene caption
- Whether the explanation changes across conditions (modality sensitivity)
- Evidence alignment score
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs" / "ablation"
OUTPUTS.mkdir(parents=True, exist_ok=True)

FAILURE_EVENTS_FILE = OUTPUTS / "failure_events.json"

# --- Hazard categories for parsing ---
HAZARD_KEYWORDS = {
    "localization degradation": ["localization", "drift", "position error", "trajectory"],
    "motion blur": ["motion blur", "blur", "blurred"],
    "visual ambiguity": ["repetitive", "featureless", "textureless", "ambiguous"],
    "geometric inconsistency": ["geometric", "reprojection", "parallax"],
    "dynamic objects": ["dynamic", "pedestrian", "vehicle", "car", "traffic"],
    "illumination change": ["lighting", "illumination", "glare", "shadow", "dark"],
    "road geometry": ["curve", "curved", "intersection", "lane"],
    "GPS degradation": ["gps", "satellite", "multipath"],
    "none detected": ["no hazard", "no immediate", "stable", "no issue"],
}


def parse_hazards(text: str) -> list[str]:
    """Extract hazard labels from VLM output text."""
    text_lower = text.lower()
    found = []
    for hazard, keywords in HAZARD_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(hazard)
                break
    return found if found else ["none detected"]


def parse_claims(text: str) -> list[str]:
    """Extract factual claims from VLM output."""
    claims = []
    claim_patterns = [
        (r"feature.*count|low.*feature|few.*feature", "low_features"),
        (r"reprojection.*error|geometric.*inconsist", "high_reprojection"),
        (r"drift|trajectory.*error|position.*error", "trajectory_drift"),
        (r"violation|unsafe|constraint.*breach", "safety_violation"),
        (r"motion.*blur|blurred", "motion_blur"),
        (r"gps|satellite|multipath", "gps_issue"),
        (r"illumination|lighting|glare|shadow", "illumination"),
        (r"curve|curved|intersection|lane.*mark", "road_geometry"),
    ]
    text_lower = text.lower()
    for pattern, label in claim_patterns:
        if re.search(pattern, text_lower):
            claims.append(label)
    return claims


def build_prompt(condition: str, metrics: dict, scene_description: str = "") -> str:
    """Build the VLM prompt for a given ablation condition."""
    base = (
        "You are analyzing a driving scene for autonomous vehicle localization. "
        "Based on the available information, provide:\n"
        "1. A scene description\n"
        "2. Hypothesized causes of any localization degradation\n"
        "3. Safety hazards present in the scene\n\n"
    )

    if condition == "image_only":
        # No metrics provided
        return base + (
            "Analyze this driving scene image. Identify potential localization "
            "challenges based ONLY on what you see in the image.\n\n"
            "Scene description:\n"
            "Hypothesized localization causes:\n"
            "Safety hazards:\n"
        )
    elif condition == "metrics_only":
        # No image - only metrics
        m = metrics
        return base + (
            f"Available metrics (no image provided):\n"
            f"- ATE RMSE: {m['ate']:.2f} m\n"
            f"- Final drift: {m['drift']:.2f} m\n"
            f"- Reprojection error: {m['reproj']:.2f} px\n"
            f"- Feature count: {m['features']:.0f}\n"
            f"- STL violations: {m['violations']}\n"
            f"- Tracking quality: {m['quality']:.2f}\n\n"
            "Analyze these metrics. Identify potential localization causes and safety hazards.\n\n"
            "Scene description:\n"
            "Hypothesized localization causes:\n"
            "Safety hazards:\n"
        )
    elif condition == "combined":
        m = metrics
        return base + (
            f"Scene image is provided above.\n\n"
            f"Available metrics:\n"
            f"- ATE RMSE: {m['ate']:.2f} m\n"
            f"- Final drift: {m['drift']:.2f} m\n"
            f"- Reprojection error: {m['reproj']:.2f} px\n"
            f"- Feature count: {m['features']:.0f}\n"
            f"- STL violations: {m['violations']}\n"
            f"- Tracking quality: {m['quality']:.2f}\n\n"
            "Analyze the scene image and metrics together.\n\n"
            "Scene description:\n"
            "Hypothesized localization causes:\n"
            "Safety hazards:\n"
        )
    elif condition == "contradictory":
        m = metrics
        # Inflate metrics
        fake_ate = m["ate"] * 10.0
        fake_drift = m["drift"] * 5.0
        return base + (
            f"Scene image is provided above.\n\n"
            f"Available metrics (NOTE: these may be unreliable):\n"
            f"- ATE RMSE: {fake_ate:.2f} m\n"
            f"- Final drift: {fake_drift:.2f} m\n"
            f"- Reprojection error: {m['reproj']:.2f} px\n"
            f"- Feature count: {m['features']:.0f}\n"
            f"- STL violations: 0\n"
            f"- Tracking quality: {m['quality']:.2f}\n\n"
            "Analyze the scene image. Cross-check whether the metrics are "
            "consistent with what you observe in the image.\n\n"
            "Scene description:\n"
            "Hypothesized localization causes:\n"
            "Safety hazards:\n"
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def run_qwen_ablation(
    model, processor, failure_events: list[dict], conditions: list[str]
) -> list[dict]:
    """Run Qwen2.5-VL on failure events under each ablation condition."""
    results = []

    for i, event in enumerate(failure_events):
        seq_id = event["sequence_id"]
        frame_idx = event["frame_idx"]
        image_path = event["image_path"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot load image {image_path}: {e}")
            continue

        # Metrics from the event
        metrics = {
            "ate": event.get("pos_error_m", 1.0),
            "drift": event.get("frame_drift_m", 0.1),
            "reproj": 1.5,  # placeholder
            "features": 400.0,  # placeholder
            "violations": 0,
            "quality": event.get("tracking_quality", 0.9),
        }

        for condition in conditions:
            cache_key = f"{seq_id}_frame{frame_idx}_{condition}"
            cache_file = OUTPUTS / f"qwen_{cache_key}.json"

            if cache_file.exists():
                entry = json.loads(cache_file.read_text())
                results.append(entry)
                print(f"  [{i+1:3d}/{len(failure_events)}] {seq_id[-8:]} f{frame_idx:04d} {condition:15s} CACHED")
                continue

            prompt = build_prompt(condition, metrics)

            # Prepare inputs
            if condition == "metrics_only":
                # No image - text only
                messages = [{"role": "user", "content": prompt}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], return_tensors="pt").to(model.device)
            else:
                # With image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            # Generate
            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,
                )
            latency_s = time.time() - t0

            # Decode only new tokens
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True)

            # Parse
            hazards = parse_hazards(response)
            claims = parse_claims(response)

            entry = {
                "sequence_id": seq_id,
                "frame_idx": frame_idx,
                "image_path": str(image_path),
                "condition": condition,
                "explanation": response,
                "hazards": hazards,
                "claims": claims,
                "claim_count": len(claims),
                "latency_s": round(latency_s, 2),
                "pos_error_m": event.get("pos_error_m", 0),
                "tracking_quality": event.get("tracking_quality", 0),
            }

            cache_file.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            results.append(entry)

            short_resp = response[:80].replace("\n", " ")
            print(
                f"  [{i+1:3d}/{len(failure_events)}] {seq_id[-8:]} f{frame_idx:04d} "
                f"{condition:15s} {latency_s:5.1f}s hazards={hazards} claims={len(claims)}"
            )

            # Free GPU memory between calls
            del inputs, output_ids
            torch.cuda.empty_cache()

    return results


def main():
    print("=" * 80)
    print("Qwen2.5-VL Modality Ablation")
    print("=" * 80)

    # Load failure events
    if not FAILURE_EVENTS_FILE.exists():
        print(f"ERROR: {FAILURE_EVENTS_FILE} not found. Run select_failure_events.py first.")
        sys.exit(1)

    failure_events = json.loads(FAILURE_EVENTS_FILE.read_text())
    print(f"Loaded {len(failure_events)} failure events")

    # Load model
    print("\nLoading Qwen2.5-VL-3B-Instruct (int4 quantized)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    conditions = ["image_only", "metrics_only", "combined", "contradictory"]

    # Run ablation
    print(f"\nRunning ablation on {len(failure_events)} events x {len(conditions)} conditions...")
    results = run_qwen_ablation(model, processor, failure_events, conditions)

    # Save all results
    all_file = OUTPUTS / "qwen_ablation_all.json"
    all_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} results to {all_file}")

    # --- Summary analysis ---
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")

    # Group by condition
    by_condition = {}
    for r in results:
        c = r["condition"]
        if c not in by_condition:
            by_condition[c] = []
        by_condition[c].append(r)

    for cond in conditions:
        entries = by_condition.get(cond, [])
        if not entries:
            continue
        unique_hazards = set()
        for e in entries:
            unique_hazards.update(e["hazards"])
        mean_claims = sum(e["claim_count"] for e in entries) / len(entries)
        mean_latency = sum(e["latency_s"] for e in entries) / len(entries)
        print(f"\n  {cond:15s}: {len(entries)} events, "
              f"mean claims={mean_claims:.1f}, "
              f"mean latency={mean_latency:.1f}s, "
              f"hazards={unique_hazards}")

    # Cross-condition comparison
    print(f"\n{'=' * 80}")
    print("Cross-condition analysis (same event, different inputs)")
    print(f"{'=' * 80}")

    by_event = {}
    for r in results:
        key = f"{r['sequence_id']}_frame{r['frame_idx']}"
        if key not in by_event:
            by_event[key] = {}
        by_event[key][r["condition"]] = r

    agree = 0
    disagree = 0
    for event_key, conds in by_event.items():
        if len(conds) < 2:
            continue
        hazard_sets = {c: set(conds[c]["hazards"]) for c in conds}
        unique_sets = set(frozenset(v) for v in hazard_sets.values())
        if len(unique_sets) == 1:
            agree += 1
        else:
            disagree += 1
            seq_short = event_key.split("_drive_")[1].split("_")[0]
            frame = event_key.split("frame")[1]
            print(f"  {seq_short} f{frame}: " +
                  ", ".join(f"{c}={hazard_sets[c]}" for c in sorted(hazard_sets)))

    total = agree + disagree
    if total > 0:
        print(f"\n  Hazard agreement rate: {agree}/{total} ({100*agree/total:.0f}%)")
        print(f"  Unique explanations per condition:")
        for cond in conditions:
            entries = by_condition.get(cond, [])
            unique_exps = set(e["explanation"][:100] for e in entries)
            print(f"    {cond:15s}: {len(unique_exps)} unique out of {len(entries)}")


if __name__ == "__main__":
    main()
