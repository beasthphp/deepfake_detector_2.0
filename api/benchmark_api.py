from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from api.config import APIConfig
from api.main import create_app
from api.services.inference_service import InferenceService


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "phase2_samples" / "input"
OUTPUT_DIR = ROOT / "reports" / "phase3"
OUTPUT_JSON = OUTPUT_DIR / "api_benchmark.json"
OUTPUT_REPORT = OUTPUT_DIR / "API_BENCHMARK.md"


BENCHMARK_CASES = {
    "one_face": "one_clear_real.jpg",
    "multiple_faces": "multiple_faces_real_fake.jpg",
    "no_face": "no_face_geometric.jpg",
}


def run_benchmark(repeat_count: int = 10) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = APIConfig()
    memory_before = current_memory_mb()
    cold_started = time.perf_counter()
    service = InferenceService.load(config)
    cold_start_ms = elapsed_ms(cold_started)
    memory_after = current_memory_mb()

    payload: dict[str, Any] = {
        "environment": {
            "runtime": "Python 3.11.9",
            "device": service.device,
            "model_version": config.model_version,
            "detector": "opencv_haar_frontalface_default",
            "crop_strategy": config.selected_crop_strategy,
        },
        "cold_start": {
            "model_and_detector_load_ms": cold_start_ms,
            "process_memory_before_mb": memory_before,
            "process_memory_after_mb": memory_after,
            "process_memory_delta_mb": round(memory_after - memory_before, 3)
            if memory_before is not None and memory_after is not None
            else None,
        },
        "warmup": None,
        "cases": {},
        "sequential_warm_requests": {},
    }

    with TestClient(create_app(config, service=service)) as client:
        payload["warmup"] = post_image(client, "one_clear_real.jpg")
        for case_name, filename in BENCHMARK_CASES.items():
            payload["cases"][case_name] = post_image(client, filename)

        sequential = [post_image(client, "one_clear_fake.jpg") for _ in range(repeat_count)]
        payload["sequential_warm_requests"] = summarize_requests(sequential)
        payload["sequential_warm_requests"]["runs"] = sequential

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_REPORT.write_text(build_report(payload), encoding="utf-8")
    return payload


def post_image(client: TestClient, filename: str) -> dict[str, Any]:
    path = SAMPLE_DIR / filename
    started = time.perf_counter()
    with path.open("rb") as handle:
        response = client.post("/predict", files={"file": (filename, handle.read(), "image/jpeg")})
    client_wall_ms = elapsed_ms(started)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    return {
        "filename": filename,
        "status_code": response.status_code,
        "client_wall_ms": client_wall_ms,
        "response": body,
    }


def summarize_requests(requests: list[dict[str, Any]]) -> dict[str, Any]:
    wall_times = [float(row["client_wall_ms"]) for row in requests]
    totals = [
        float(row["response"].get("timing_ms", {}).get("total", 0.0))
        for row in requests
        if isinstance(row.get("response"), dict)
    ]
    failed = [row for row in requests if row["status_code"] >= 400]
    return {
        "count": len(requests),
        "failed_count": len(failed),
        "client_wall_ms_avg": round(statistics.mean(wall_times), 3) if wall_times else None,
        "client_wall_ms_median": round(statistics.median(wall_times), 3) if wall_times else None,
        "client_wall_ms_min": round(min(wall_times), 3) if wall_times else None,
        "client_wall_ms_max": round(max(wall_times), 3) if wall_times else None,
        "response_total_ms_avg": round(statistics.mean(totals), 3) if totals else None,
        "response_total_ms_median": round(statistics.median(totals), 3) if totals else None,
    }


def build_report(payload: dict[str, Any]) -> str:
    cold = payload["cold_start"]
    sequential = payload["sequential_warm_requests"]
    lines = [
        "# API Benchmark",
        "",
        "## Cold Startup",
        "",
        f"- Model and detector load: {cold['model_and_detector_load_ms']:.3f} ms",
        f"- Memory before load: {fmt_optional(cold['process_memory_before_mb'])} MB",
        f"- Memory after load: {fmt_optional(cold['process_memory_after_mb'])} MB",
        f"- Memory delta: {fmt_optional(cold['process_memory_delta_mb'])} MB",
        "",
        "## Warm Requests",
        "",
        "| Case | HTTP | API status | Faces | Decode ms | Detection ms | Crop ms | Classification ms | Serialization ms | Response total ms | Client wall ms |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_name, case in payload["cases"].items():
        response = case["response"]
        timing = response.get("timing_ms", {}) if isinstance(response, dict) else {}
        lines.append(
            "| {case} | {http} | {status} | {faces} | {decode} | {detect} | {crop} | {classify} | {serial} | {total} | {wall} |".format(
                case=case_name,
                http=case["status_code"],
                status=response.get("status", "error") if isinstance(response, dict) else "error",
                faces=response.get("faces_detected", "") if isinstance(response, dict) else "",
                decode=fmt_optional(timing.get("decode")),
                detect=fmt_optional(timing.get("face_detection")),
                crop=fmt_optional(timing.get("crop_preprocessing")),
                classify=fmt_optional(timing.get("classification")),
                serial=fmt_optional(timing.get("serialization")),
                total=fmt_optional(timing.get("total")),
                wall=fmt_optional(case["client_wall_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## Ten Sequential Warm Requests",
            "",
            f"- Count: {sequential['count']}",
            f"- Failed: {sequential['failed_count']}",
            f"- Average client wall time: {fmt_optional(sequential['client_wall_ms_avg'])} ms",
            f"- Median client wall time: {fmt_optional(sequential['client_wall_ms_median'])} ms",
            f"- Min client wall time: {fmt_optional(sequential['client_wall_ms_min'])} ms",
            f"- Max client wall time: {fmt_optional(sequential['client_wall_ms_max'])} ms",
            f"- Average response total time: {fmt_optional(sequential['response_total_ms_avg'])} ms",
            f"- Median response total time: {fmt_optional(sequential['response_total_ms_median'])} ms",
            "",
            "Full benchmark JSON is saved in `reports/phase3/api_benchmark.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def current_memory_mb() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    process = psutil.Process()
    return round(process.memory_info().rss / (1024 * 1024), 3)


def fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def main() -> int:
    payload = run_benchmark()
    print(json.dumps({"cold_start": payload["cold_start"], "sequential": payload["sequential_warm_requests"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
