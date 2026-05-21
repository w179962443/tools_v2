#!/usr/bin/env python3
"""
NSFW Filter Script

Reads every image and video file in an input directory, runs NudeNet NSFW
detection on each file, and moves detected NSFW files to separate output
directories — one for images, one for videos.

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
        detector: Detection backend — ``nudenet``, ``clip``, or ``falconsai``.
        clip_nsfw_prompts: Custom NSFW text prompts for the CLIP detector.
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

    print(f"Initializing NSFW detector ({detector})…")
    detector_cls = _DETECTOR_MAP[detector]
    if detector == "clip" and clip_nsfw_prompts:
        detector_obj = detector_cls(threshold=threshold, nsfw_prompts=clip_nsfw_prompts)
    else:
        detector_obj = detector_cls(threshold=threshold)

    img_moved = vid_moved = safe_count = error_count = 0

    # ---------------------------------------------------------------
    # Images
    # ---------------------------------------------------------------
    for idx, img_file in enumerate(image_files, 1):
        prefix = f"[img {idx:>{len(str(len(image_files)))}}/{len(image_files)}]"
        print(f"{prefix} {img_file.name}", end="", flush=True)

        try:
            result = detector_obj.detect_image(str(img_file))
            if result["is_nsfw"]:
                labels = ", ".join(result["nsfw_labels"])
                print(f"  →  NSFW ({labels})  →  MOVE")
                if not dry_run:
                    shutil.move(str(img_file), str(_safe_dest(img_out, img_file)))
                img_moved += 1
            else:
                print("  →  safe")
                safe_count += 1
        except Exception as exc:
            print(f"  →  ERROR: {exc}")
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
                    f"  →  NSFW @ {first['timestamp']:.1f}s "
                    f"(frame {first['frame_idx']})  →  MOVE"
                )
                if not dry_run:
                    shutil.move(str(vid_file), str(_safe_dest(vid_out, vid_file)))
                vid_moved += 1
            else:
                checked = result["checked_frames"]
                print(f"  →  safe  ({checked} frames checked)")
                safe_count += 1
        except Exception as exc:
            print(f"  →  ERROR: {exc}")
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
        print("(Dry-run mode — no files were moved)")


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
        help="Confidence threshold 0–1 (default: 0.5)",
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
    )


if __name__ == "__main__":
    main()
