from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vision_language_localization.vlm.rule_based_vlm import VLMExplanationResult


@dataclass
class HfLocalVLMConfig:
    model_name: str


class HfLocalVLM:
    """Optional local Hugging Face vision-language adapter.

    This backend performs image captioning and then combines visual evidence with
    localization metrics to produce a grounded scene/safety explanation.
    """

    def __init__(self, config: HfLocalVLMConfig) -> None:
        self.model_name = config.model_name
        self._processor: Any = None
        self._model: Any = None
        self._mode = "auto"
        try:
            import transformers

            auto_processor = getattr(transformers, "AutoProcessor")
            auto_model_cls = getattr(transformers, "AutoModelForVision2Seq")

            self._processor = auto_processor.from_pretrained(self.model_name)
            self._model = auto_model_cls.from_pretrained(self.model_name)
        except Exception:
            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor

                self._mode = "blip"
                self._processor = BlipProcessor.from_pretrained(self.model_name)
                self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            except Exception as exc:
                raise ImportError(
                    "Failed to initialize local VLM. Ensure compatible transformers/torch versions are installed."
                ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def explain(
        self,
        ate_rmse: float,
        drift_final_m: float,
        mean_reprojection_error: float,
        mean_feature_count: float,
        violation_count: int,
        scene_image_path: Path | None = None,
    ) -> VLMExplanationResult:
        scene_caption: str | None = None
        if scene_image_path is not None and scene_image_path.exists():
            image = Image.open(scene_image_path).convert("RGB")
            model_inputs = self._processor(images=image, return_tensors="pt")
            model_inputs = {k: v.to(self._device) for k, v in model_inputs.items()}
            generated_ids = self._model.generate(**model_inputs, max_new_tokens=32)
            scene_caption = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        visual_prefix = ""
        if scene_caption:
            visual_prefix = f"Scene observation: {scene_caption}. "

        text = (
            visual_prefix
            + "Localization analysis: "
            + f"ATE={ate_rmse:.2f} m, drift={drift_final_m:.2f} m, reprojection error={mean_reprojection_error:.2f}, "
            + f"feature count={mean_feature_count:.0f}, violations={violation_count}. "
        )

        if mean_feature_count < 500:
            text += "Low feature support may reduce matching robustness. "
        if mean_reprojection_error > 1.8:
            text += "Geometric inconsistency appears elevated. "
        if violation_count > 0:
            text += "Runtime safety constraints were violated in this segment. "

        text += "Recommendation: use conservative control when confidence drops."

        hazards: list[str] = []
        low = text.lower()
        if "risk" in low or "violate" in low:
            hazards.append("model-reported risk")
        if violation_count > 0:
            hazards.append("safety constraint violation")
        if ate_rmse > 1.5:
            hazards.append("localization degradation risk")
        if not hazards:
            hazards.append("no immediate hazard detected")

        hallucination_risk = 0.2 if mean_feature_count > 400 else 0.35
        consistency_score = 1.0 - 0.4 * hallucination_risk

        return VLMExplanationResult(
            explanation=text,
            hazards=hazards,
            hallucination_risk=float(hallucination_risk),
            consistency_score=float(consistency_score),
            scene_caption=scene_caption,
        )
