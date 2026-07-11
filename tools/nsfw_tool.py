#!/usr/bin/env python3
"""
NSFW detection workflow based on NudeNet.

Supports both images and videos.  Videos are sampled at configurable intervals
to keep runtime reasonable.

Provides both a programmatic Python API and a command-line interface.

CLI Usage:
    python -m tools.nsfw_tool --file photo.jpg
    python -m tools.nsfw_tool --file video.mp4 --sample-interval 3
    python tools/nsfw_tool.py --file photo.jpg --threshold 0.6 --output-json result.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

_NUDE_DETECTOR_CLS = None
_CV2_MODULE = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Body-part labels from NudeNet that we consider NSFW.
NSFW_LABELS: set = {
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
}

IMAGE_EXTENSIONS: set = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tiff",
    ".tif",
}

VIDEO_EXTENSIONS: set = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
}


def _require_nudenet():
    """Import NudeNet lazily so other detectors can work without it installed."""
    global _NUDE_DETECTOR_CLS
    if _NUDE_DETECTOR_CLS is None:
        try:
            from nudenet import NudeDetector as _NudeDetector
        except ImportError as exc:
            raise ImportError(
                "nudenet is required for NSFWDetector.\n"
                "Install with: pip install nudenet"
            ) from exc
        _NUDE_DETECTOR_CLS = _NudeDetector
    return _NUDE_DETECTOR_CLS


def _require_cv2():
    """Import OpenCV lazily so image-only commands do not fail at import time."""
    global _CV2_MODULE
    if _CV2_MODULE is None:
        try:
            import cv2 as _cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for video detection.\n"
                "Install with: pip install opencv-python"
            ) from exc
        _CV2_MODULE = _cv2
    return _CV2_MODULE


@contextmanager
def _detector_readable_path(file_path: str) -> Iterator[str]:
    """Yield a model-friendly path, copying to an ASCII temp path when needed.

    Some Windows/OpenCV-based model stacks fail on non-ASCII source paths.
    Copying to a short ASCII temp path avoids false read errors for files under
    directories such as ``E:\\图\\...``.
    """
    normalized_path = os.fspath(file_path)
    if normalized_path.isascii():
        yield normalized_path
        return

    suffix = Path(normalized_path).suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="nsfw_") as tmp_dir:
        temp_path = os.path.join(tmp_dir, f"input{suffix}")
        shutil.copyfile(normalized_path, temp_path)
        yield temp_path


def _load_pil_image(image_path: str):
    """Load an image through Pillow in a way that is robust to Windows paths."""
    from PIL import Image as _PILImage

    with open(image_path, "rb") as image_file:
        image = _PILImage.open(image_file)
        image.load()
    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Base class (shared video sampling + dispatch logic)
# ---------------------------------------------------------------------------


class _BaseNSFWDetector:
    """Abstract base for NSFW detectors.

    Subclasses must implement :meth:`detect_image`.  Video detection and the
    unified :meth:`detect` dispatcher are provided here so they do not need to
    be repeated in every subclass.
    """

    threshold: float

    def detect_image(self, image_path: str) -> dict:
        """Detect NSFW content in a single image.  Must be overridden."""
        raise NotImplementedError

    def detect_video(
        self,
        video_path: str,
        sample_interval: float = 2.0,
        max_frames: int = 100,
    ) -> dict:
        """Detect NSFW content in a video by sampling frames.

        Sampling stops immediately after the first NSFW frame is found.

        Args:
            video_path: Path to the video file.
            sample_interval: Seconds between sampled frames (default 2.0).
            max_frames: Maximum number of frames to check (default 100).

        Returns:
            Dict with keys:
                - 'is_nsfw' (bool)
                - 'nsfw_frames' (list[dict]): frames where NSFW was detected,
                  each with 'frame_idx', 'timestamp', 'nsfw_labels'
                - 'checked_frames' (int): number of frames actually checked
                - 'total_frames' (int): total frames in the video

        Raises:
            FileNotFoundError: If the video does not exist.
            ValueError: If the video cannot be opened.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cv2 = _require_cv2()
        with _detector_readable_path(video_path) as readable_video_path:
            cap = cv2.VideoCapture(readable_video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval: int = max(1, int(fps * sample_interval))

            nsfw_frames: List[dict] = []
            checked_frames: int = 0
            is_nsfw: bool = False

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    frame_idx = 0
                    while checked_frames < max_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret:
                            break

                        tmp_path = os.path.join(tmp_dir, "frame.jpg")
                        cv2.imwrite(tmp_path, frame)

                        try:
                            img_result = self.detect_image(tmp_path)
                            if img_result["is_nsfw"]:
                                is_nsfw = True
                                nsfw_frames.append(
                                    {
                                        "frame_idx": frame_idx,
                                        "timestamp": round(frame_idx / fps, 2),
                                        "nsfw_labels": img_result["nsfw_labels"],
                                    }
                                )
                                break
                        except Exception:
                            pass  # Skip unreadable frame

                        checked_frames += 1
                        frame_idx += frame_interval
                        if frame_idx >= total_frames:
                            break
            finally:
                cap.release()

        return {
            "is_nsfw": is_nsfw,
            "nsfw_frames": nsfw_frames,
            "checked_frames": checked_frames,
            "total_frames": total_frames,
        }

    def detect(self, file_path: str, **kwargs) -> dict:
        """Detect NSFW content in an image or video file.

        Dispatches to :meth:`detect_image` or :meth:`detect_video` based on
        the file extension.

        Args:
            file_path: Path to an image or video file.
            **kwargs: Extra keyword arguments forwarded to :meth:`detect_video`
                      (``sample_interval``, ``max_frames``).

        Returns:
            Detection result dict with an additional 'file_type' key
            ('image' or 'video').

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = Path(file_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            result = self.detect_image(file_path)
            result["file_type"] = "image"
        elif ext in VIDEO_EXTENSIONS:
            result = self.detect_video(file_path, **kwargs)
            result["file_type"] = "video"
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported images: {sorted(IMAGE_EXTENSIONS)}. "
                f"Supported videos: {sorted(VIDEO_EXTENSIONS)}."
            )
        result["file_path"] = file_path
        return result


# ---------------------------------------------------------------------------
# NudeNet detector
# ---------------------------------------------------------------------------


class NSFWDetector(_BaseNSFWDetector):
    """NSFW detector using NudeNet.

    Supports images and videos.  For videos, frames are sampled at a
    configurable interval; detection stops at the first NSFW frame found.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize NSFW detector.

        Args:
            threshold: Minimum confidence score (0-1) for a detection to be
                       counted as NSFW.  Default 0.5.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold
        self._detector = _require_nudenet()()

    # ------------------------------------------------------------------
    # Image detection
    # ------------------------------------------------------------------

    def detect_image(self, image_path: str) -> dict:
        """Detect NSFW content in a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with keys:
                - 'is_nsfw' (bool)
                - 'nsfw_labels' (list[str]): unique NSFW label names found
                - 'nsfw_detections' (list[dict]): filtered detections above threshold
                - 'detections' (list[dict]): all raw NudeNet detections

        Raises:
            FileNotFoundError: If the image does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with _detector_readable_path(image_path) as readable_image_path:
            raw_detections: List[dict] = self._detector.detect(readable_image_path)

        nsfw_detections = [
            d
            for d in raw_detections
            if d.get("class") in NSFW_LABELS and d.get("score", 0.0) >= self.threshold
        ]
        max_nsfw_score = max(
            (
                float(d.get("score", 0.0))
                for d in raw_detections
                if d.get("class") in NSFW_LABELS
            ),
            default=0.0,
        )

        return {
            "is_nsfw": len(nsfw_detections) > 0,
            "nsfw_labels": sorted({d["class"] for d in nsfw_detections}),
            "nsfw_detections": nsfw_detections,
            "detections": raw_detections,
            "max_nsfw_score": round(max_nsfw_score, 4),
        }


# ---------------------------------------------------------------------------
# CLIP zero-shot detector
# ---------------------------------------------------------------------------


class CLIPDetector(_BaseNSFWDetector):
    """NSFW detector using OpenAI CLIP zero-shot image-text matching.

    Matches an image against a configurable list of normal and NSFW text
    prompts.  A softmax is computed over all prompts; the image is flagged
    as NSFW when the **maximum** probability among all NSFW prompts reaches
    ``threshold``.

    Requires: ``openai-clip`` and ``torch``
        pip install openai-clip torch
    """

    DEFAULT_NORMAL_PROMPTS: List[str] = [
        "a normal everyday photo",
        "a safe for work image",
        "a landscape photograph",
        "a family friendly picture",
    ]
    DEFAULT_NSFW_PROMPTS: List[str] = [
        "nude person",
        "explicit sexual content",
        "pornographic image",
        "naked body exposed",
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        normal_prompts: List[str] = None,
        nsfw_prompts: List[str] = None,
        device: str = None,
    ) -> None:
        """Initialize the CLIP detector.

        Args:
            threshold: Maximum NSFW-prompt probability (0-1) above which the
                       image is considered NSFW.  Default 0.5.  Recommended
                       0.65-0.80 for balanced precision/recall.
            normal_prompts: Override the default "safe" text prompts.
            nsfw_prompts: Override the default NSFW text prompts.
            device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
                    Auto-detected when omitted.
        """
        try:
            import clip as _clip
            import torch as _torch
        except ImportError:
            raise ImportError(
                "openai-clip and torch are required for CLIPDetector.\n"
                "Install with: pip install openai-clip torch"
            )

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold
        self._normal_prompts = normal_prompts or self.DEFAULT_NORMAL_PROMPTS
        self._nsfw_prompts = nsfw_prompts or self.DEFAULT_NSFW_PROMPTS
        self._all_prompts = self._normal_prompts + self._nsfw_prompts

        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self._device = device
        self._torch = _torch
        self._clip = _clip

        self._model, self._preprocess = _clip.load("ViT-B/32", device=device)
        self._model.eval()

    def detect_image(self, image_path: str) -> dict:
        """Detect NSFW content in a single image using CLIP.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with keys:
                - 'is_nsfw' (bool)
                - 'nsfw_labels' (list[str]): top NSFW prompts that contributed
                - 'scores' (dict[str, float]): probability per prompt

        Raises:
            FileNotFoundError: If the image does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = self._preprocess(_load_pil_image(image_path)).unsqueeze(0).to(
            self._device
        )
        text_tokens = self._clip.tokenize(self._all_prompts).to(self._device)

        with self._torch.no_grad():
            logits_per_image, _ = self._model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1)[0].tolist()

        n_normal = len(self._normal_prompts)
        nsfw_probs = probs[n_normal:]
        nsfw_max = max(nsfw_probs) if nsfw_probs else 0.0

        # Report NSFW prompts that individually contributed meaningfully.
        nsfw_labels = [
            prompt
            for prompt, score in sorted(
                zip(self._nsfw_prompts, nsfw_probs),
                key=lambda x: x[1],
                reverse=True,
            )
            if score >= 0.05
        ][:3]

        scores = {
            prompt: round(probs[i], 4) for i, prompt in enumerate(self._all_prompts)
        }

        return {
            "is_nsfw": nsfw_max >= self.threshold,
            "nsfw_labels": nsfw_labels,
            "scores": scores,
            "max_nsfw_score": round(nsfw_max, 4),
        }


# ---------------------------------------------------------------------------
# Falconsai / HuggingFace ViT detector
# ---------------------------------------------------------------------------


class FalconsaiDetector(_BaseNSFWDetector):
    """NSFW detector using ``Falconsai/nsfw_image_detection`` on HuggingFace.

    Uses a ViT-based image classification model with **two** output labels:
    ``normal`` and ``nsfw``.

    An image is considered NSFW when the ``nsfw`` label has a confidence
    score >= ``threshold``.

    Requires: ``transformers`` and ``torch``
        pip install transformers torch
    """

    NSFW_LABELS: set = {"nsfw"}

    def __init__(
        self,
        threshold: float = 0.5,
        device: str = None,
    ) -> None:
        """Initialize the Falconsai detector.

        Args:
            threshold: Minimum score (0-1) for a non-normal label to trigger
                       an NSFW result.  Default 0.5.
            device: HuggingFace pipeline device (``"cpu"``, ``"cuda"``, or
                    device index integer as string).  Auto-detected when omitted.
        """
        try:
            from transformers import pipeline as _hf_pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for FalconsaiDetector.\n"
                "Install with: pip install transformers torch"
            )

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold

        hf_device: int
        if device is not None:
            hf_device = int(device) if device.lstrip("-").isdigit() else -1
        else:
            try:
                import torch as _torch

                hf_device = 0 if _torch.cuda.is_available() else -1
            except ImportError:
                hf_device = -1

        self._classifier = _hf_pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=hf_device,
        )

    def detect_image(self, image_path: str) -> dict:
        """Detect NSFW content in a single image using Falconsai ViT.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with keys:
                - 'is_nsfw' (bool)
                - 'nsfw_labels' (list[str]): non-normal labels above threshold
                - 'scores' (dict[str, float]): confidence per label

        Raises:
            FileNotFoundError: If the image does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_results = self._classifier(_load_pil_image(image_path))
        scores = {r["label"]: round(r["score"], 4) for r in raw_results}

        nsfw_detections = [
            r
            for r in raw_results
            if r["label"].lower() in self.NSFW_LABELS and r["score"] >= self.threshold
        ]
        nsfw_labels = [
            r["label"]
            for r in sorted(nsfw_detections, key=lambda x: x["score"], reverse=True)
        ]
        max_nsfw_score = max(
            (
                float(r["score"])
                for r in raw_results
                if r["label"].lower() in self.NSFW_LABELS
            ),
            default=0.0,
        )

        return {
            "is_nsfw": len(nsfw_detections) > 0,
            "nsfw_labels": nsfw_labels,
            "scores": scores,
            "max_nsfw_score": round(max_nsfw_score, 4),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DETECTOR_MAP = {
    "nudenet": NSFWDetector,
    "clip": CLIPDetector,
    "falconsai": FalconsaiDetector,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSFW detection tool (NudeNet / CLIP / Falconsai)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.nsfw_tool --file photo.jpg
  python -m tools.nsfw_tool --file photo.jpg --detector clip
  python -m tools.nsfw_tool --file photo.jpg --detector falconsai
  python -m tools.nsfw_tool --file video.mp4 --sample-interval 5 --max-frames 50
  python -m tools.nsfw_tool --file photo.jpg --output-json result.json
  python -m tools.nsfw_tool --file photo.jpg --status-only
        """,
    )

    parser.add_argument("--file", required=True, help="Path to an image or video file")
    parser.add_argument(
        "--detector",
        choices=list(_DETECTOR_MAP),
        default="nudenet",
        help="Detection backend to use (default: nudenet)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=2.0,
        metavar="SECS",
        help="Seconds between sampled video frames (default: 2.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum video frames to check (default: 100)",
    )
    parser.add_argument(
        "--output-json",
        metavar="FILE",
        help="Save full detection results to a JSON file",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Exit with code 1 if NSFW, 0 if safe (useful for shell scripting)",
    )

    args = parser.parse_args()

    detector_cls = _DETECTOR_MAP[args.detector]
    detector = detector_cls(threshold=args.threshold)
    result = detector.detect(
        args.file,
        sample_interval=args.sample_interval,
        max_frames=args.max_frames,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_json}")

    if args.status_only:
        sys.exit(1 if result["is_nsfw"] else 0)

    status = "NSFW" if result["is_nsfw"] else "SAFE"
    print(f"File   : {args.file}")
    print(f"Type   : {result['file_type']}")
    print(f"Status : {status}")

    if result["file_type"] == "image" and result.get("nsfw_labels"):
        print(f"Labels : {', '.join(result['nsfw_labels'])}")

    if result["file_type"] == "video":
        print(f"Frames checked : {result['checked_frames']} / {result['total_frames']}")
        if result.get("nsfw_frames"):
            for fr in result["nsfw_frames"][:5]:
                ts = fr["timestamp"]
                labels = ", ".join(fr["nsfw_labels"])
                print(f"  NSFW @ {ts:.1f}s (frame {fr['frame_idx']}): {labels}")


if __name__ == "__main__":
    main()
