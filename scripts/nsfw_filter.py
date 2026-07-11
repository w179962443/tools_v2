#!/usr/bin/env python3
"""
NSFW Filter Script

Reads every image and video file in an input directory, runs NudeNet NSFW
detection on each file, and moves detected NSFW files to separate output
directories - one for images, one for videos.

Usage examples:
  python scripts/nsfw_filter.py --input-dir ./media --output-images ./nsfw_images \
      --output-videos ./nsfw_videos
  python scripts/nsfw_filter.py --input-dir ./media --output-images ./img \
      --output-videos ./vid --dry-run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.nsfw_tool import (
    NSFWDetector,
    CLIPDetector,
    FalconsaiDetector,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)  # noqa: E402

_DETECTOR_MAP = {
    "nudenet": NSFWDetector,
    "clip": CLIPDetector,
    "falconsai": FalconsaiDetector,
}


def _build_detector(
    detector_name: str,
    threshold: float,
    clip_nsfw_prompts: list = None,
):
    """Instantiate a detector, reusing CLIP prompt overrides when applicable."""
    detector_cls = _DETECTOR_MAP[detector_name]
    if detector_name == "clip" and clip_nsfw_prompts:
        return detector_cls(threshold=threshold, nsfw_prompts=clip_nsfw_prompts)
    return detector_cls(threshold=threshold)


def _score_text(result: dict) -> str:
    """Format the highest NSFW score for progress output."""
    score = result.get("max_nsfw_score")
    if score is None:
        return "n/a"
    return f"{score:.3f}"


def _safe_dest(dest_dir: Path, src: Path) -> Path:
    """Return a collision-free destination path inside dest_dir."""
    dest = dest_dir / src.name
    if not dest.exists():
        return dest
    counter = 1
    while True:
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        if not dest.exists():
            return dest
        counter += 1


def process_directory(
    input_dir: str,
    output_images_dir: str,
    output_videos_dir: str,
    threshold: float = 0.5,
    sample_interval: float = 2.0,
    max_frames: int = 100,
    dry_run: bool = False,
    detector: str = "nudenet",
    clip_nsfw_prompts: list = None,
    secondary_detector: Optional[str] = None,
    secondary_threshold: Optional[float] = None,
    review_score: Optional[float] = None,
) -> None:
    """Scan input_dir and move NSFW files to the appropriate output directory.

    Args:
        input_dir: Directory containing source media files.
        output_images_dir: Destination for NSFW images.
        output_videos_dir: Destination for NSFW videos.
        threshold: Detection confidence threshold (default 0.5).
        sample_interval: Seconds between sampled video frames (default 2.0).
        max_frames: Maximum frames to inspect per video (default 100).
        dry_run: Report actions without moving files.
        detector: Detection backend - ``nudenet``, ``clip``, or ``falconsai``.
        clip_nsfw_prompts: Custom NSFW text prompts for the CLIP detector.
        secondary_detector: Optional second-pass detector for borderline images.
        secondary_threshold: Threshold for the second-pass detector.
        review_score: Re-check safe images when their max NSFW score is still
            at least this value.
    """
    input_path = Path(input_dir).expanduser().resolve()
    img_out = Path(output_images_dir).expanduser().resolve()
    vid_out = Path(output_videos_dir).expanduser().resolve()

    if not input_path.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(f for f in input_path.rglob("*") if f.is_file())
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS]
    video_files = [f for f in all_files if f.suffix.lower() in VIDEO_EXTENSIONS]

    total = len(image_files) + len(video_files)
    if total == 0:
        print(f"No image or video files found in: {input_dir}")
        return

    print(
        f"Found {len(image_files)} image(s) and {len(video_files)} video(s)  "
        f"|  threshold: {threshold}  |  detector: {detector}"
    )

    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        vid_out.mkdir(parents=True, exist_ok=True)

    print(f"Initializing NSFW detector ({detector})...")
    detector_obj = _build_detector(
        detector_name=detector,
        threshold=threshold,
        clip_nsfw_prompts=clip_nsfw_prompts,
    )

    secondary_obj = None
    secondary_threshold_value = secondary_threshold
    if secondary_detector:
        secondary_threshold_value = (
            threshold if secondary_threshold is None else secondary_threshold
        )
        print(
            f"Initializing second-pass detector ({secondary_detector})..."
            f"  |  threshold: {secondary_threshold_value}"
            f"  |  review score: {review_score if review_score is not None else 'disabled'}"
        )
        secondary_obj = _build_detector(
            detector_name=secondary_detector,
            threshold=secondary_threshold_value,
            clip_nsfw_prompts=clip_nsfw_prompts,
        )

    img_moved = vid_moved = safe_count = error_count = 0

    # ---------------------------------------------------------------
    # Images
    # ---------------------------------------------------------------
    for idx, img_file in enumerate(image_files, 1):
        prefix = f"[img {idx:>{len(str(len(image_files)))}}/{len(image_files)}]"
        print(f"{prefix} {img_file.name}", end="", flush=True)

        try:
            primary_result = detector_obj.detect_image(str(img_file))
            primary_score = primary_result.get("max_nsfw_score", 0.0) or 0.0

            result = primary_result
            decision_detector = detector
            reviewed = False

            if (
                secondary_obj is not None
                and not primary_result["is_nsfw"]
                and review_score is not None
                and primary_score >= review_score
            ):
                reviewed = True
                secondary_result = secondary_obj.detect_image(str(img_file))
                if secondary_result["is_nsfw"]:
                    result = secondary_result
                    decision_detector = secondary_detector

            if result["is_nsfw"]:
                labels = ", ".join(result["nsfw_labels"]) or "nsfw"
                if reviewed and decision_detector == secondary_detector:
                    print(
                        f"  ->  NSFW via {decision_detector} after review "
                        f"(primary score={primary_score:.3f}; labels={labels})  ->  MOVE"
                    )
                else:
                    print(
                        f"  ->  NSFW via {decision_detector} "
                        f"(score={_score_text(result)}; labels={labels})  ->  MOVE"
                    )
                if not dry_run:
                    shutil.move(str(img_file), str(_safe_dest(img_out, img_file)))
                img_moved += 1
            else:
                if reviewed:
                    print(
                        f"  ->  safe after {secondary_detector} review "
                        f"(primary score={primary_score:.3f})"
                    )
                else:
                    print(f"  ->  safe (score={_score_text(primary_result)})")
                safe_count += 1
        except Exception as exc:
            print(f"  ->  ERROR: {exc}")
            error_count += 1

    # ---------------------------------------------------------------
    # Videos
    # ---------------------------------------------------------------
    for idx, vid_file in enumerate(video_files, 1):
        prefix = f"[vid {idx:>{len(str(len(video_files)))}}/{len(video_files)}]"
        print(f"{prefix} {vid_file.name}", end="", flush=True)

        try:
            result = detector_obj.detect_video(
                str(vid_file),
                sample_interval=sample_interval,
                max_frames=max_frames,
            )
            if result["is_nsfw"]:
                first = result["nsfw_frames"][0]
                print(
                    f"  ->  NSFW @ {first['timestamp']:.1f}s "
                    f"(frame {first['frame_idx']})  ->  MOVE"
                )
                if not dry_run:
                    shutil.move(str(vid_file), str(_safe_dest(vid_out, vid_file)))
                vid_moved += 1
            else:
                checked = result["checked_frames"]
                print(f"  ->  safe  ({checked} frames checked)")
                safe_count += 1
        except Exception as exc:
            print(f"  ->  ERROR: {exc}")
            error_count += 1

    print()
    print(
        f"Done.  "
        f"NSFW images moved: {img_moved}  |  "
        f"NSFW videos moved: {vid_moved}  |  "
        f"Safe: {safe_count}  |  "
        f"Errors: {error_count}"
    )
    if dry_run:
        print("(Dry-run mode - no files were moved)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and separate NSFW images/videos into output directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default NudeNet backend
  python scripts/nsfw_filter.py \\
      --input-dir "C:\\Users\\Administrator\\Documents\\Tencent Files\\179962443" \\
      --output-images ~/nsfw_images \\
      --output-videos ~/nsfw_videos

  # CLIP zero-shot backend
  python scripts/nsfw_filter.py \\
      --input-dir ./media \\
      --output-images ./nsfw_images \\
      --output-videos ./nsfw_videos \\
      --detector clip

  # Falconsai HuggingFace ViT backend
  python scripts/nsfw_filter.py \\
      --input-dir ./media \\
      --output-images ./nsfw_images \\
      --output-videos ./nsfw_videos \\
      --detector falconsai --threshold 0.6

  # CLIP with custom NSFW prompts
  python scripts/nsfw_filter.py \\
      --input-dir ./media \\
      --output-images ./nsfw_images \\
      --output-videos ./nsfw_videos \\
      --detector clip \\
      --clip-nsfw-prompts "stockings,bare legs,bikini"

  # Falconsai first pass + CLIP second pass for borderline safe images
  python scripts/nsfw_filter.py \\
      --input-dir ./media \\
      --output-images ./nsfw_images \\
      --output-videos ./nsfw_videos \\
      --detector falconsai --threshold 0.5 \\
      --secondary-detector clip --secondary-threshold 0.62 \\
      --review-score 0.2

  # Dry-run
  python scripts/nsfw_filter.py \\
      --input-dir ./media \\
      --output-images ./nsfw_images \\
      --output-videos ./nsfw_videos \\
      --dry-run
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Source directory containing image and/or video files",
    )
    parser.add_argument(
        "--output-images",
        required=True,
        metavar="DIR",
        help="Destination directory for NSFW images",
    )
    parser.add_argument(
        "--output-videos",
        required=True,
        metavar="DIR",
        help="Destination directory for NSFW videos",
    )
    parser.add_argument(
        "--detector",
        choices=["nudenet", "clip", "falconsai"],
        default="nudenet",
        help="Detection backend: nudenet (default), clip, or falconsai",
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
        help="Maximum video frames to inspect per file (default: 100)",
    )
    parser.add_argument(
        "--clip-nsfw-prompts",
        metavar="PROMPTS",
        default=None,
        help=(
            "Comma-separated NSFW text prompts for the CLIP detector "
            '(e.g. "stockings,bare legs,bikini"). '
            "Only used when --detector clip is set."
        ),
    )
    parser.add_argument(
        "--secondary-detector",
        choices=["nudenet", "clip", "falconsai"],
        default=None,
        help=(
            "Optional second-pass detector for borderline images that were "
            "classified as safe by the primary detector"
        ),
    )
    parser.add_argument(
        "--secondary-threshold",
        type=float,
        default=None,
        help=(
            "Confidence threshold for --secondary-detector. "
            "Defaults to the primary --threshold."
        ),
    )
    parser.add_argument(
        "--review-score",
        type=float,
        default=None,
        help=(
            "When a safe image still has max_nsfw_score >= this value, run "
            "the second-pass detector on it."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without moving any files",
    )

    args = parser.parse_args()

    clip_prompts = None
    if args.clip_nsfw_prompts:
        clip_prompts = [
            p.strip() for p in args.clip_nsfw_prompts.split(",") if p.strip()
        ]

    process_directory(
        input_dir=args.input_dir,
        output_images_dir=args.output_images,
        output_videos_dir=args.output_videos,
        threshold=args.threshold,
        sample_interval=args.sample_interval,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
        detector=args.detector,
        clip_nsfw_prompts=clip_prompts,
        secondary_detector=args.secondary_detector,
        secondary_threshold=args.secondary_threshold,
        review_score=args.review_score,
    )


if __name__ == "__main__":
    main()
