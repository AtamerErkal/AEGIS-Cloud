"""
AEGIS-Cloud — Realistic Edge Video Test
========================================
Run from project root:
    python test_video_edge.py --video data/sim_samples/test_video.mp4
    python test_video_edge.py --video data/sim_samples/test_video.mp4 --moondream
    python test_video_edge.py --video data/sim_samples/test_video.mp4 --fps 10

Tests the full edge stack realistically:
  1. ONNX YOLOv8-nano inference per frame
  2. NATO Incident Report logging
  3. Best-frame selection (5s window)
  4. Optional Moondream VLM reasoning
  5. AIOps metrics: FPS, latency, CPU, RAM

Usage on Jetson:
    # Transfer video first (from Windows PowerShell):
    #   scp test_video.mp4 atamer@<jetson-ip>:~/aegis_project/data/sim_samples/
    #
    # Then on Jetson:
    #   cd ~/aegis_project && conda activate aegis
    #   python test_video_edge.py --video data/sim_samples/test_video.mp4 --moondream
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

# Project root bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from edge.src.perception.vision_node import VisionNode, BestFrameSelector


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS Edge Video Test")
    p.add_argument(
        "--video",
        default="data/videos/test_drone.mp4",
        help="Path to test video file (default: data/videos/test_drone.mp4)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target processing FPS (default: 5)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="YOLO confidence threshold (default: 0.30 — lowered for maritime/ship targets)",
    )
    p.add_argument(
        "--moondream",
        action="store_true",
        help="Enable Moondream VLM reasoning on best frames",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = full video)",
    )
    p.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated frames to data/logs/snapshots/annotated/",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Annotated frame writer
# ──────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, detections: list, frame_id: int) -> np.ndarray:
    """Draw bounding boxes + labels on the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    color_map = {
        "Hostile": (0, 0, 255),    # Red
        "Unknown": (0, 165, 255),  # Orange
        "Friendly": (0, 255, 0),  # Green
    }
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        color = color_map.get(det.risk_level, (255, 255, 0))
        cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)
        label = f"{det.target_type.upper()} {det.confidence:.2f} [{det.risk_level}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (px1, py1 - th - 6), (px1 + tw, py1), color, -1)
        cv2.putText(out, label, (px1, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(out, f"AEGIS | Frame {frame_id}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main test runner
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    video_path = Path(args.video)

    print("=" * 70)
    print("  AEGIS-Cloud — Realistic Edge Video Test")
    print("=" * 70)

    # ── Pre-flight checks ──────────────────────────────────────────────────
    if not video_path.exists():
        print(f"\n[ERROR] Video not found: {video_path}")
        print("\nTransfer it from Windows (PowerShell):")
        print(f"  scp test_drone.mp4 atamer@<jetson-ip>:~/aegis_project/data/videos/test_drone.mp4")
        print("\nOr generate a synthetic test video:")
        print("  python -c \"")
        print("    import cv2, numpy as np")
        print("    out = cv2.VideoWriter('data/sim_samples/test_video.mp4',")
        print("          cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))")
        print("    [out.write(np.random.randint(0,255,(480,640,3),dtype=np.uint8)) for _ in range(300)]")
        print("    out.release()\"")
        sys.exit(1)

    # Get video metadata
    probe = cv2.VideoCapture(str(video_path))
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = probe.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    duration_s   = total_frames / source_fps if source_fps > 0 else 0

    print(f"\n  Video      : {video_path}")
    print(f"  Resolution : {width}x{height}")
    print(f"  Duration   : {duration_s:.1f}s  ({total_frames} frames @ {source_fps:.1f} fps)")
    print(f"  Test FPS   : {args.fps}  (processing every {source_fps/args.fps:.1f} source frames)")
    print(f"  Conf thresh: {args.conf}")
    print(f"  Moondream  : {'ON' if args.moondream else 'OFF'}")
    if args.max_frames:
        print(f"  Max frames : {args.max_frames}")
    print()

    # ── Patch config for this test run ────────────────────────────────────
    # Override sim_video_path and conf in-memory via a custom config dict
    import yaml
    cfg_path = Path("edge/config/edge_settings.yaml")
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)

    cfg["simulation_mode"] = True
    cfg["inference"]["sim_video_path"] = str(video_path)
    cfg["inference"]["confidence_threshold"] = args.conf
    cfg["inference"]["target_fps"] = args.fps

    # Write temp config
    tmp_cfg = Path("edge/config/_test_video_tmp.yaml")
    with tmp_cfg.open("w") as fh:
        yaml.dump(cfg, fh)

    # ── Load reasoning node (optional) ───────────────────────────────────
    reasoning = None
    if args.moondream:
        from edge.src.perception.reasoning_node import ReasoningNode
        reasoning = ReasoningNode(config_path=tmp_cfg)
        print(f"[INFO] Moondream loaded — endpoint={reasoning.endpoint}  model={reasoning.model}")

    # ── Init VisionNode ───────────────────────────────────────────────────
    print("[INFO] Initialising VisionNode (loading ONNX model)...")
    t_init = time.perf_counter()
    node = VisionNode(config_path=tmp_cfg, reasoning_node=reasoning)
    init_ms = (time.perf_counter() - t_init) * 1000
    print(f"[INFO] VisionNode ready in {init_ms:.0f}ms")

    health = node.health_check()
    print(f"[INFO] Health: {health['status']} | model_loaded={health['model_loaded']} "
          f"| capture_open={health['capture_open']}")
    print()

    # ── Annotated output dir ──────────────────────────────────────────────
    annotated_dir = Path("data/logs/snapshots/annotated")
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Annotated frames → {annotated_dir}")

    # ── Statistics ────────────────────────────────────────────────────────
    stats = {
        "frames_processed": 0,
        "frames_with_detections": 0,
        "total_detections": 0,
        "class_counts": defaultdict(int),
        "latencies_ms": [],
        "moondream_calls": 0,
        "moondream_total_ms": 0.0,
    }

    best_frame_sel = BestFrameSelector(window_s=cfg["inference"].get("best_frame_window_s", 5.0))

    print("─" * 70)
    print(f"{'Frame':>7}  {'Detections':<30}  {'Latency':>8}  {'Moondream'}")
    print("─" * 70)

    t_run_start = time.perf_counter()

    try:
        for frame_idx, detections in enumerate(node.run(), start=1):
            t_frame = time.perf_counter()
            stats["frames_processed"] += 1

            if args.max_frames and frame_idx > args.max_frames:
                break

            # Per-frame latency (approximate — from generator tick)
            latency_ms = (t_frame - t_run_start) * 1000 / frame_idx
            stats["latencies_ms"].append(latency_ms)

            if detections:
                stats["frames_with_detections"] += 1
                stats["total_detections"] += len(detections)
                for d in detections:
                    stats["class_counts"][d.target_type] += 1

                det_summary = ", ".join(
                    f"{d.target_type}({d.confidence:.2f})" for d in detections
                )
                moondream_note = ""

                # Moondream reasoning on best frame
                if reasoning:
                    best = best_frame_sel.update(
                        node._next_frame() or np.zeros((1, 1, 3), dtype=np.uint8),
                        detections,
                    )
                    if best is not None:
                        bf, bd = best
                        det = max(bd, key=lambda d: d.confidence)
                        t_md = time.perf_counter()
                        result = reasoning.describe(
                            frame=bf,
                            bbox=det.bbox,
                            detection_id=f"f{frame_idx}",
                        )
                        md_ms = (time.perf_counter() - t_md) * 1000
                        stats["moondream_calls"] += 1
                        stats["moondream_total_ms"] += md_ms
                        moondream_note = f"[MD {md_ms:.0f}ms] {result.description[:60]}…"

                # Annotate and save frame
                if args.save_annotated:
                    frame_data = node._cap.read()[1] if node._cap else None
                    if frame_data is not None:
                        ann = annotate_frame(frame_data, detections, frame_idx)
                        cv2.imwrite(str(annotated_dir / f"frame_{frame_idx:05d}.jpg"), ann)

                print(f"{frame_idx:>7}  {det_summary:<30}  {latency_ms:>7.1f}ms  {moondream_note}")

            elif frame_idx % 25 == 0:
                # Heartbeat on empty frames
                elapsed = time.perf_counter() - t_run_start
                real_fps = frame_idx / elapsed
                print(f"{frame_idx:>7}  [no detections]                  "
                      f"{latency_ms:>7.1f}ms  FPS: {real_fps:.1f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by operator (Ctrl+C).")
    finally:
        tmp_cfg.unlink(missing_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_run_start
    real_fps = stats["frames_processed"] / elapsed_total if elapsed_total > 0 else 0

    print("\n" + "=" * 70)
    print("  AEGIS TEST SUMMARY")
    print("=" * 70)
    print(f"  Frames processed    : {stats['frames_processed']}")
    print(f"  Frames w/ detections: {stats['frames_with_detections']} "
          f"({100*stats['frames_with_detections']/max(stats['frames_processed'],1):.1f}%)")
    print(f"  Total detections    : {stats['total_detections']}")
    if stats["class_counts"]:
        print(f"  By class            : " +
              ", ".join(f"{k}={v}" for k, v in sorted(stats["class_counts"].items())))
    print(f"  Real FPS            : {real_fps:.2f}")
    print(f"  Total test time     : {elapsed_total:.1f}s")
    if args.moondream and stats["moondream_calls"] > 0:
        avg_md = stats["moondream_total_ms"] / stats["moondream_calls"]
        print(f"  Moondream calls     : {stats['moondream_calls']}")
        print(f"  Moondream avg lat   : {avg_md:.0f}ms")
    if args.save_annotated:
        print(f"  Annotated frames    → {annotated_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
