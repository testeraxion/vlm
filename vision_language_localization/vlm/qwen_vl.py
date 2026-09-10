from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import certifi
import torch
from PIL import Image

from vision_language_localization.vlm.rule_based_vlm import VLMExplanationResult


@dataclass
class QwenVLMConfig:
    model_name: str
    max_new_tokens: int = 180
    temperature: float = 0.0


class QwenVLVLM:
    """Qwen multimodal backend for image-conditioned safety/localization analysis."""

    def __init__(self, config: QwenVLMConfig) -> None:
        self.model_name = config.model_name
        self.max_new_tokens = max(32, config.max_new_tokens)
        self.temperature = max(0.0, float(config.temperature))

        self._processor: Any = None
        self._model: Any = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use certifi's CA bundle explicitly to avoid broken inherited SSL_CERT_FILE
        # values from external environments (e.g., base conda).
        ca_bundle = certifi.where()
        os.environ["SSL_CERT_FILE"] = ca_bundle
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["CURL_CA_BUNDLE"] = ca_bundle

        try:
            import transformers

            auto_processor = getattr(transformers, "AutoProcessor")
            auto_model = getattr(transformers, "AutoModelForImageTextToText")
            self._processor = auto_processor.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = auto_model.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as exc:
            raise ImportError(
                "Failed to initialize Qwen backend. This is often due to SSL/certificate or model download issues. "
                "Ensure .venv has dependencies installed with: pip install -r requirements_vlm_local.txt"
            ) from exc

        self._model.to(self._device)

    def _generate_multimodal(self, image_path: Path, prompt_text: str) -> str:
        image = Image.open(image_path).convert("RGB")

        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            chat_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self._processor(
                text=[chat_text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        else:
            model_inputs = self._processor(images=image, text=prompt_text, return_tensors="pt")

        model_inputs = {k: v.to(self._device) for k, v in model_inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature

        generated_ids = self._model.generate(**model_inputs, **generate_kwargs)

        if "input_ids" in model_inputs:
            prefix_len = model_inputs["input_ids"].shape[-1]
            trimmed = generated_ids[:, prefix_len:]
        else:
            trimmed = generated_ids

        output = self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if not output:
            return ""
        return str(output[0]).strip()

    def explain(
        self,
        ate_rmse: float,
        drift_final_m: float,
        mean_reprojection_error: float,
        mean_feature_count: float,
        violation_count: int,
        scene_image_path: Path | None = None,
    ) -> VLMExplanationResult:
        if scene_image_path is None or not scene_image_path.exists():
            fallback = (
                "No scene image available. Localization metrics indicate "
                f"ATE={ate_rmse:.2f}m, drift={drift_final_m:.2f}m, reprojection={mean_reprojection_error:.2f}, "
                f"features={mean_feature_count:.0f}, violations={violation_count}."
            )
            return VLMExplanationResult(
                explanation=fallback,
                hazards=["insufficient visual context"],
                hallucination_risk=0.45,
                consistency_score=0.75,
                scene_caption=None,
            )

        prompt = (
            "Analyze this driving scene and localization state. "
            "1) Describe the scene briefly. "
            "2) Explain likely localization degradation causes grounded in visual clues and metrics. "
            "3) Mention safety hazards and uncertainty. "
            "Keep response concise. "
            f"Metrics: ATE={ate_rmse:.3f}, drift={drift_final_m:.3f}, reprojection_error={mean_reprojection_error:.3f}, "
            f"feature_count={mean_feature_count:.1f}, violations={violation_count}."
        )

        generated = self._generate_multimodal(scene_image_path, prompt)
        lines = [x.strip() for x in generated.split(".") if x.strip()]
        scene_caption = lines[0] if lines else None

        hazards: list[str] = []
        low = generated.lower()
        if "hazard" in low or "risk" in low or "unsafe" in low:
            hazards.append("model-reported risk")
        if violation_count > 0:
            hazards.append("safety constraint violation")
        if mean_feature_count < 500:
            hazards.append("weak visual observability")
        if ate_rmse > 1.5:
            hazards.append("localization degradation risk")
        if not hazards:
            hazards.append("no immediate hazard detected")

        hallucination_risk = 0.15
        if mean_feature_count < 300:
            hallucination_risk += 0.15
        if "uncertain" not in low:
            hallucination_risk += 0.1

        consistency_score = max(0.0, 1.0 - 0.4 * hallucination_risk)

        return VLMExplanationResult(
            explanation=generated,
            hazards=hazards,
            hallucination_risk=float(min(1.0, hallucination_risk)),
            consistency_score=float(consistency_score),
            scene_caption=scene_caption,
        )
