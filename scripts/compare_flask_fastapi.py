"""Manual black-box regression comparison; this script never starts either server."""
import argparse
import json
from pathlib import Path

import httpx

NUMERIC_KEYS = ("confidence", "real_score", "fake_score", "base_score_fake", "local_score_fake", "threshold")


def request_prediction(base_url: str, video_path: Path) -> dict:
    with video_path.open("rb") as stream:
        response = httpx.post(f"{base_url.rstrip('/')}/predict-video", files={"video": (video_path.name, stream, "video/mp4")}, timeout=1200)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--flask-url", default="http://127.0.0.1:5001")
    parser.add_argument("--fastapi-url", default="http://127.0.0.1:5000")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    old, new = request_prediction(args.flask_url, args.video), request_prediction(args.fastapi_url, args.video)
    failures = []
    for key in ("prediction", "label", "status", "frames_used"):
        if old.get(key) != new.get(key): failures.append(f"{key}: {old.get(key)!r} != {new.get(key)!r}")
    for key in NUMERIC_KEYS:
        if old.get(key) is None and new.get(key) is None: continue
        if abs(float(old.get(key)) - float(new.get(key))) > args.tolerance: failures.append(f"{key}: {old.get(key)} != {new.get(key)}")
    if old.get("feature_debug", {}).get("feature_vector_shape") != new.get("feature_debug", {}).get("feature_vector_shape"):
        failures.append("feature vector shape berbeda")
    print(json.dumps({"compatible": not failures, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

