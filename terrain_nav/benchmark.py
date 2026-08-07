"""Offline benchmark runner for profile-vector localization variants."""

from __future__ import annotations

import csv
import json
import math
import time
import zipfile
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.sax.saxutils import escape

import numpy as np

from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform, normalize_heading
from terrain_nav.matcher import Candidate, ProfileMatcher
from terrain_nav.metrics import ProfileComparison
from terrain_nav.profile import extract_profile, rotate_offsets
from terrain_nav.sensors import Measurement, SensorSimulator
from terrain_nav.terrain import TerrainManager

ProgressCallback = Callable[[str], None]
StopCallback = Callable[[], bool]


@dataclass(frozen=True)
class BenchmarkVariant:
    """One profile-vector strategy to evaluate."""

    name: str
    point_count: int
    mode: str


@dataclass(frozen=True)
class BenchmarkRoute:
    """A route expressed as an in-map start point plus absolute heading segments."""

    name: str
    start_x: float
    start_y: float
    segments: tuple[tuple[float, float], ...]

    @property
    def total_distance_m(self) -> float:
        return float(sum(distance for _heading, distance in self.segments))


@dataclass(frozen=True)
class BenchmarkSample:
    distance_m: float
    x: float
    y: float
    heading_deg: float
    measurement: Measurement


@dataclass(frozen=True)
class BenchmarkProfile:
    distances_m: np.ndarray
    offsets: list[tuple[float, float, float]]
    laser_agl: np.ndarray
    laser_valid: np.ndarray
    baro_msl: np.ndarray
    true_current_x: float
    true_current_y: float
    true_current_heading_deg: float


@dataclass(frozen=True)
class BenchmarkSummary:
    route_name: str
    checkpoint_fraction: float
    checkpoint_distance_m: float
    route_total_distance_m: float
    variant_name: str
    vector_mode: str
    requested_points: int
    used_points: int
    profile_span_m: float
    fix_accepted: bool
    wrong_fix: bool
    candidate_found: bool
    position_error_m: float | None
    score: float | None
    inlier_rmse_m: float | None
    correlation: float | None
    valid_ratio: float | None
    elapsed_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    summaries: list[BenchmarkSummary]
    route_paths: dict[str, list[tuple[float, float]]]
    route_headings: dict[str, float]
    best_profile: ProfileComparison | None = None
    summary_csv_path: str | None = None
    details_jsonl_path: str | None = None
    excel_path: str | None = None


ROUTE_TEMPLATES: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = (
    ("duz_dogu", ((90.0, 1.0),)),
    ("duz_kuzey", ((0.0, 1.0),)),
    ("l_dogu_kuzey", ((90.0, 0.55), (0.0, 0.45))),
    ("l_kuzey_dogu", ((0.0, 0.55), (90.0, 0.45))),
    ("zikzak", ((90.0, 0.30), (0.0, 0.23), (270.0, 0.30), (0.0, 0.17))),
    (
        "merdiven",
        ((90.0, 0.20), (0.0, 0.20), (90.0, 0.20), (0.0, 0.20), (270.0, 0.20)),
    ),
    ("capraz_kesit", ((45.0, 0.50), (315.0, 0.50))),
    ("u_donus", ((90.0, 0.35), (0.0, 0.18), (270.0, 0.35), (0.0, 0.12))),
)

DEFAULT_POINT_COUNTS = (50, 100, 150, 250, 500, 750, 1000)
DEFAULT_CHECKPOINT_FRACTIONS = (0.35, 0.50, 0.65, 0.80, 1.00)
DEFAULT_ROUTE_ORIENTATIONS = (0.0, 45.0, 90.0, 135.0)
DEFAULT_ROUTE_CENTER_FRACTIONS = (
    (0.50, 0.50),
    (0.32, 0.32),
    (0.68, 0.32),
    (0.32, 0.68),
    (0.68, 0.68),
    (0.50, 0.28),
    (0.50, 0.72),
    (0.28, 0.50),
    (0.72, 0.50),
)
DEFAULT_ROUTE_SCALE_FACTORS = (0.55, 0.75, 1.00)
DEFAULT_WRONG_FIX_THRESHOLD_M = 50.0


def default_benchmark_variants(
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
) -> list[BenchmarkVariant]:
    variants = []
    for point_count in point_counts:
        count = max(2, int(point_count))
        variants.append(BenchmarkVariant(f"raw_{count}", count, "raw"))
        variants.append(BenchmarkVariant(f"interp_{count}", count, "interp"))
    return variants


def build_benchmark_routes(
    terrain: TerrainManager,
    *,
    max_routes: int | None = None,
    route_orientations: Sequence[float] = DEFAULT_ROUTE_ORIENTATIONS,
    route_center_fractions: Sequence[tuple[float, float]] = DEFAULT_ROUTE_CENTER_FRACTIONS,
    route_scale_factors: Sequence[float] = DEFAULT_ROUTE_SCALE_FACTORS,
) -> list[BenchmarkRoute]:
    """Build a diverse route suite scaled to the current map extent."""
    width_m, height_m = terrain.get_extent()
    target_distance_m = min(2600.0, max(40.0, min(width_m, height_m) * 0.68))
    margin_x = width_m * 0.08
    margin_y = height_m * 0.08
    routes: list[BenchmarkRoute] = []
    route_index = 0

    for template_index, (name, normalized_segments) in enumerate(ROUTE_TEMPLATES):
        for orientation_index, orientation_deg in enumerate(route_orientations):
            if max_routes is not None and len(routes) >= max_routes:
                return routes
            scale_factor = route_scale_factors[
                (template_index + orientation_index) % len(route_scale_factors)
            ]
            center_fraction = route_center_fractions[route_index % len(route_center_fractions)]
            routes.append(
                _build_route(
                    name,
                    normalized_segments,
                    orientation_deg=float(orientation_deg),
                    center_fraction=center_fraction,
                    scale_factor=float(scale_factor),
                    target_distance_m=target_distance_m,
                    width_m=width_m,
                    height_m=height_m,
                    margin_x=margin_x,
                    margin_y=margin_y,
                )
            )
            route_index += 1

    return routes


def _build_route(
    template_name: str,
    normalized_segments: Sequence[tuple[float, float]],
    *,
    orientation_deg: float,
    center_fraction: tuple[float, float],
    scale_factor: float,
    target_distance_m: float,
    width_m: float,
    height_m: float,
    margin_x: float,
    margin_y: float,
) -> BenchmarkRoute:
    rotated_segments = tuple(
        (normalize_heading(heading_deg + orientation_deg), distance_fraction)
        for heading_deg, distance_fraction in normalized_segments
    )

    points = [(0.0, 0.0)]
    cursor_x = 0.0
    cursor_y = 0.0
    for heading_deg, fraction in rotated_segments:
        dx, dy = _offset_meters(float(fraction), heading_deg)
        cursor_x += dx
        cursor_y += dy
        points.append((cursor_x, cursor_y))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    path_width = max(xs) - min(xs)
    path_height = max(ys) - min(ys)
    fit_limits = []
    available_width = max(1.0, width_m - 2.0 * margin_x)
    available_height = max(1.0, height_m - 2.0 * margin_y)
    if path_width > 0.0:
        fit_limits.append(available_width / path_width)
    if path_height > 0.0:
        fit_limits.append(available_height / path_height)
    fit_distance_m = min(fit_limits) if fit_limits else target_distance_m
    scale_m = min(target_distance_m * max(0.10, scale_factor), fit_distance_m)

    scaled_points = [(x * scale_m, y * scale_m) for x, y in points]
    scaled_xs = [point[0] for point in scaled_points]
    scaled_ys = [point[1] for point in scaled_points]
    half_width = (max(scaled_xs) - min(scaled_xs)) / 2.0
    half_height = (max(scaled_ys) - min(scaled_ys)) / 2.0
    desired_center_x = width_m * center_fraction[0]
    desired_center_y = -height_m * center_fraction[1]

    center_x = _clamp_center(
        desired_center_x,
        lower=margin_x + half_width,
        upper=width_m - margin_x - half_width,
        fallback=width_m / 2.0,
    )
    center_y = _clamp_center(
        desired_center_y,
        lower=-height_m + margin_y + half_height,
        upper=-margin_y - half_height,
        fallback=-height_m / 2.0,
    )

    path_center_x = (min(scaled_xs) + max(scaled_xs)) / 2.0
    path_center_y = (min(scaled_ys) + max(scaled_ys)) / 2.0
    start_x = center_x - path_center_x
    start_y = center_y - path_center_y
    orientation_label = int(round(orientation_deg)) % 360
    scale_label = int(round(scale_factor * 100))
    center_label = (
        f"c{int(round(center_fraction[0] * 100)):02d}_"
        f"{int(round(center_fraction[1] * 100)):02d}"
    )

    return BenchmarkRoute(
        name=f"{template_name}_o{orientation_label:03d}_{center_label}_s{scale_label}",
        start_x=start_x,
        start_y=start_y,
        segments=tuple(
            (heading_deg, float(fraction) * scale_m) for heading_deg, fraction in rotated_segments
        ),
    )


def _clamp_center(value: float, *, lower: float, upper: float, fallback: float) -> float:
    if lower > upper:
        return fallback
    return min(max(value, lower), upper)


def run_benchmark_suite(
    config: LocalizationConfig,
    *,
    point_counts: Sequence[int] = DEFAULT_POINT_COUNTS,
    checkpoint_fractions: Sequence[float] = DEFAULT_CHECKPOINT_FRACTIONS,
    max_routes: int = 32,
    dense_sample_count: int | None = None,
    wrong_fix_threshold_m: float = DEFAULT_WRONG_FIX_THRESHOLD_M,
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
    output_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Run multi-route, multi-checkpoint profile-vector benchmarks."""
    variants = default_benchmark_variants(point_counts)
    checkpoints = _normalized_checkpoint_fractions(checkpoint_fractions)
    dense_count = dense_sample_count or max(1200, max(variant.point_count for variant in variants))
    dense_count = max(2, int(dense_count))

    terrain = TerrainManager(config.terrain)
    try:
        config = _with_safe_constant_altitude(config, terrain)
        algorithm = _benchmark_algorithm(config, terrain.nav_dem.shape)
        config = dataclass_replace(config, algorithm=algorithm)
        ct = CoordinateTransform(terrain.dx, terrain.dy)
        matcher = ProfileMatcher(config, terrain.get_navigation_dem(copy=False), ct)
        routes = build_benchmark_routes(terrain, max_routes=max_routes)

        summaries: list[BenchmarkSummary] = []
        route_paths: dict[str, list[tuple[float, float]]] = {}
        route_headings: dict[str, float] = {}
        best_profile: ProfileComparison | None = None
        best_error = float("inf")
        total_jobs = len(routes) * len(checkpoints) * len(variants)
        completed_jobs = 0
        if progress_callback is not None:
            progress_callback(
                f"[BENCH] Plan: {len(routes)} rota x {len(checkpoints)} kontrol "
                f"x {len(variants)} vektör = {total_jobs} koşu"
            )

        for route_index, route in enumerate(routes):
            if _should_stop(stop_requested):
                break
            samples = _simulate_route_samples(
                config,
                terrain,
                route,
                dense_sample_count=dense_count,
                seed_offset=route_index,
            )
            route_paths[route.name] = _thin_path([(sample.x, sample.y) for sample in samples])
            route_headings[route.name] = samples[-1].heading_deg if samples else 0.0

            for checkpoint_fraction in checkpoints:
                if _should_stop(stop_requested):
                    break
                checkpoint_samples = _samples_until_fraction(samples, checkpoint_fraction)
                for variant in variants:
                    if _should_stop(stop_requested):
                        break
                    profile = _profile_for_variant(checkpoint_samples, variant)
                    summary, comparison = _evaluate_profile(
                        config,
                        matcher,
                        ct,
                        route,
                        checkpoint_fraction,
                        variant,
                        profile,
                        wrong_fix_threshold_m=wrong_fix_threshold_m,
                    )
                    summaries.append(summary)
                    completed_jobs += 1
                    if (
                        summary.fix_accepted
                        and not summary.wrong_fix
                        and summary.position_error_m is not None
                        and summary.position_error_m < best_error
                    ):
                        best_error = summary.position_error_m
                        best_profile = comparison
                    if progress_callback is not None:
                        status = "fix" if summary.fix_accepted else "yok"
                        if summary.wrong_fix:
                            status = "yanlış"
                        error_text = (
                            f"{summary.position_error_m:.1f} m"
                            if summary.position_error_m is not None
                            else "-"
                        )
                        progress_callback(
                            f"[BENCH {completed_jobs}/{total_jobs}] {route.name} "
                            f"@{checkpoint_fraction:.0%} / {variant.name}: "
                            f"{status}, hata {error_text}, {summary.elapsed_ms:.0f} ms"
                        )

        result = BenchmarkResult(
            summaries=summaries,
            route_paths=route_paths,
            route_headings=route_headings,
            best_profile=best_profile,
        )
        if output_dir is not None:
            result = save_benchmark_result(result, output_dir)
        return result
    finally:
        terrain.close()


def save_benchmark_result(result: BenchmarkResult, output_dir: str | Path) -> BenchmarkResult:
    """Persist benchmark summaries as CSV and JSONL."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"benchmark_{stamp}_summary.csv"
    details_path = out_dir / f"benchmark_{stamp}_details.jsonl"
    excel_path = out_dir / f"benchmark_{stamp}.xlsx"

    rows = [asdict(summary) for summary in result.summaries]
    fieldnames = list(rows[0].keys()) if rows else list(BenchmarkSummary.__dataclass_fields__)
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with details_path.open("w", encoding="utf-8") as jsonl_file:
        for summary in result.summaries:
            jsonl_file.write(json.dumps(asdict(summary), ensure_ascii=False) + "\n")

    saved_result = BenchmarkResult(
        summaries=result.summaries,
        route_paths=result.route_paths,
        route_headings=result.route_headings,
        best_profile=result.best_profile,
        summary_csv_path=str(summary_path),
        details_jsonl_path=str(details_path),
        excel_path=str(excel_path),
    )
    write_benchmark_excel(saved_result, excel_path)
    return saved_result


def _normalized_checkpoint_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    cleaned = sorted({max(0.01, min(1.0, float(fraction))) for fraction in fractions})
    if not cleaned:
        return (1.0,)
    if cleaned[-1] < 1.0:
        cleaned.append(1.0)
    return tuple(cleaned)


def benchmark_overview_rows(result: BenchmarkResult) -> list[dict[str, object]]:
    """Return high-level metrics for UI and Excel reporting."""
    summaries = result.summaries
    total = len(summaries)
    fixed = sum(1 for summary in summaries if summary.fix_accepted)
    wrong = sum(1 for summary in summaries if summary.wrong_fix)
    errors = _position_errors(summaries)
    elapsed_values = [summary.elapsed_ms for summary in summaries]
    best = _best_summary(summaries)
    rows: list[dict[str, object]] = [
        {"metric": "Rota sayisi", "value": len(result.route_paths)},
        {
            "metric": "Kontrol noktasi sayisi",
            "value": len({summary.checkpoint_fraction for summary in summaries}),
        },
        {"metric": "Toplam kosu", "value": total},
        {"metric": "Fix sayisi", "value": fixed},
        {"metric": "Fix orani (%)", "value": _percent(fixed, total)},
        {"metric": "Yanlis fix sayisi", "value": wrong},
        {"metric": "Yanlis fix orani (%)", "value": _percent(wrong, total)},
        {"metric": "Medyan konum hatasi (m)", "value": _median(errors)},
        {"metric": "P95 konum hatasi (m)", "value": _percentile(errors, 95)},
        {"metric": "Ortalama sure (ms)", "value": _mean(elapsed_values)},
    ]
    if best is not None:
        rows.extend(
            [
                {"metric": "En iyi varyant", "value": best.variant_name},
                {"metric": "En iyi rota", "value": best.route_name},
                {"metric": "En iyi hata (m)", "value": best.position_error_m},
            ]
        )
    if result.excel_path:
        rows.append({"metric": "Excel dosyasi", "value": result.excel_path})
    if result.summary_csv_path:
        rows.append({"metric": "CSV dosyasi", "value": result.summary_csv_path})
    if result.details_jsonl_path:
        rows.append({"metric": "JSONL dosyasi", "value": result.details_jsonl_path})
    return rows


def benchmark_variant_summary_rows(result: BenchmarkResult) -> list[dict[str, object]]:
    """Aggregate benchmark results by vector/profile variant."""
    rows = []
    for variant_name in _ordered_variant_names(result.summaries):
        variant_rows = [
            summary for summary in result.summaries if summary.variant_name == variant_name
        ]
        first = variant_rows[0]
        rows.append(
            _aggregate_row(
                variant_rows,
                {
                    "variant": variant_name,
                    "mode": first.vector_mode,
                    "points": first.requested_points,
                },
            )
        )
    return rows


def benchmark_route_summary_rows(result: BenchmarkResult) -> list[dict[str, object]]:
    """Aggregate benchmark results by route."""
    rows = []
    for route_name in _ordered_route_names(result.summaries):
        route_rows = [summary for summary in result.summaries if summary.route_name == route_name]
        first = route_rows[0]
        best = _best_summary(route_rows)
        row = _aggregate_row(
            route_rows,
            {
                "route": route_name,
                "distance_m": first.route_total_distance_m,
                "best_variant": best.variant_name if best is not None else "",
            },
        )
        rows.append(row)
    rows.sort(key=lambda row: (_sort_optional_number(row["median_error_m"]), -int(row["fix_count"])))
    return rows


def benchmark_checkpoint_summary_rows(result: BenchmarkResult) -> list[dict[str, object]]:
    """Aggregate benchmark results by route-progress checkpoint."""
    rows = []
    checkpoints = sorted({summary.checkpoint_fraction for summary in result.summaries})
    for checkpoint in checkpoints:
        checkpoint_rows = [
            summary for summary in result.summaries if summary.checkpoint_fraction == checkpoint
        ]
        rows.append(
            _aggregate_row(
                checkpoint_rows,
                {
                    "checkpoint_pct": checkpoint * 100.0,
                    "checkpoint_fraction": checkpoint,
                },
            )
        )
    return rows


def write_benchmark_excel(result: BenchmarkResult, path: str | Path) -> None:
    """Write a multi-sheet Excel workbook without requiring optional dependencies."""
    detail_rows = [asdict(summary) for summary in result.summaries]
    sheets = [
        ("Genel Ozet", benchmark_overview_rows(result)),
        ("Vektor Ozet", benchmark_variant_summary_rows(result)),
        ("Rota Ozet", benchmark_route_summary_rows(result)),
        ("Kontrol Ozet", benchmark_checkpoint_summary_rows(result)),
        ("Detaylar", detail_rows),
    ]
    _write_xlsx(Path(path), sheets)


def format_benchmark_summary(result: BenchmarkResult, *, limit: int = 8) -> str:
    """Format a compact operator-facing result summary."""
    if not result.summaries:
        return "Benchmark sonucu yok."

    total = len(result.summaries)
    fixed = sum(1 for summary in result.summaries if summary.fix_accepted)
    wrong = sum(1 for summary in result.summaries if summary.wrong_fix)
    checkpoint_count = len({summary.checkpoint_fraction for summary in result.summaries})
    lines = [
        f"Benchmark: {len(result.route_paths)} rota, {checkpoint_count} kontrol, "
        f"{total} koşu, fix {fixed}/{total}, yanlış {wrong}"
    ]

    accepted = [
        summary
        for summary in result.summaries
        if summary.fix_accepted and not summary.wrong_fix and summary.position_error_m is not None
    ]
    if accepted:
        best = min(accepted, key=lambda summary: summary.position_error_m or float("inf"))
        lines.append(
            f"En iyi: {best.variant_name} / {best.route_name} "
            f"{best.position_error_m:.1f} m, RMSE {_format_optional(best.inlier_rmse_m)}"
        )

    for variant_name in _ordered_variant_names(result.summaries)[:limit]:
        variant_rows = [summary for summary in result.summaries if summary.variant_name == variant_name]
        variant_fixed = [summary for summary in variant_rows if summary.fix_accepted]
        errors = [
            float(summary.position_error_m)
            for summary in variant_fixed
            if summary.position_error_m is not None
        ]
        median_error = float(np.median(errors)) if errors else None
        p95_error = float(np.percentile(errors, 95)) if errors else None
        variant_wrong = sum(1 for summary in variant_rows if summary.wrong_fix)
        mean_ms = float(np.mean([summary.elapsed_ms for summary in variant_rows]))
        lines.append(
            f"{variant_name}: fix {len(variant_fixed)}/{len(variant_rows)}, "
            f"medyan {_format_optional(median_error)}, "
            f"P95 {_format_optional(p95_error)}, yanlış {variant_wrong}, {mean_ms:.0f} ms"
        )

    if result.excel_path:
        lines.append(f"Excel: {result.excel_path}")
    if result.summary_csv_path:
        lines.append(f"CSV: {result.summary_csv_path}")
    return "\n".join(lines)


def _ordered_variant_names(summaries: Iterable[BenchmarkSummary]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for summary in summaries:
        if summary.variant_name not in seen:
            seen.add(summary.variant_name)
            names.append(summary.variant_name)
    return names


def _ordered_route_names(summaries: Iterable[BenchmarkSummary]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for summary in summaries:
        if summary.route_name not in seen:
            seen.add(summary.route_name)
            names.append(summary.route_name)
    return names


def _aggregate_row(
    rows: Sequence[BenchmarkSummary],
    prefix: dict[str, object],
) -> dict[str, object]:
    total = len(rows)
    fixed = sum(1 for summary in rows if summary.fix_accepted)
    wrong = sum(1 for summary in rows if summary.wrong_fix)
    candidate_count = sum(1 for summary in rows if summary.candidate_found)
    errors = _position_errors(rows)
    rmse_values = [
        summary.inlier_rmse_m for summary in rows if summary.inlier_rmse_m is not None
    ]
    corr_values = [summary.correlation for summary in rows if summary.correlation is not None]
    elapsed_values = [summary.elapsed_ms for summary in rows]
    return {
        **prefix,
        "runs": total,
        "candidate_count": candidate_count,
        "fix_count": fixed,
        "fix_rate_pct": _percent(fixed, total),
        "wrong_fix_count": wrong,
        "wrong_fix_rate_pct": _percent(wrong, total),
        "median_error_m": _median(errors),
        "p95_error_m": _percentile(errors, 95),
        "best_error_m": min(errors) if errors else None,
        "mean_inlier_rmse_m": _mean(rmse_values),
        "mean_correlation": _mean(corr_values),
        "mean_elapsed_ms": _mean(elapsed_values),
    }


def _position_errors(summaries: Iterable[BenchmarkSummary]) -> list[float]:
    return [
        float(summary.position_error_m)
        for summary in summaries
        if summary.fix_accepted and summary.position_error_m is not None
    ]


def _best_summary(summaries: Iterable[BenchmarkSummary]) -> BenchmarkSummary | None:
    accepted = [
        summary
        for summary in summaries
        if summary.fix_accepted and not summary.wrong_fix and summary.position_error_m is not None
    ]
    if not accepted:
        return None
    return min(accepted, key=lambda summary: summary.position_error_m or float("inf"))


def _percent(part: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(part) * 100.0 / float(total)


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return None
    return float(np.mean(clean))


def _median(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return None
    return float(np.median(clean))


def _percentile(values: Sequence[float | None], percentile: float) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return None
    return float(np.percentile(clean, percentile))


def _sort_optional_number(value: object) -> float:
    if value is None:
        return float("inf")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return numeric if np.isfinite(numeric) else float("inf")


def _write_xlsx(path: Path, sheets: Sequence[tuple[str, list[dict[str, object]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _xlsx_content_types(len(sheets)))
        archive.writestr("_rels/.rels", _xlsx_root_relationships())
        archive.writestr("xl/workbook.xml", _xlsx_workbook_xml(sheets))
        archive.writestr("xl/_rels/workbook.xml.rels", _xlsx_workbook_relationships(len(sheets)))
        archive.writestr("xl/styles.xml", _xlsx_styles_xml())
        for index, (_sheet_name, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _xlsx_sheet_xml(rows))


def _xlsx_content_types(sheet_count: int) -> str:
    sheet_overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{sheet_overrides}</Types>"
    )


def _xlsx_root_relationships() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )


def _xlsx_workbook_xml(sheets: Sequence[tuple[str, list[dict[str, object]]]]) -> str:
    sheet_xml = "\n".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _rows) in enumerate(sheets, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheet_xml}</sheets></workbook>"
    )


def _xlsx_workbook_relationships(sheet_count: int) -> str:
    relationships = [
        (
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        for index in range(1, sheet_count + 1)
    ]
    relationships.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "\n".join(relationships)
        + "</Relationships>"
    )


def _xlsx_styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/>'
        '<bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" '
        'applyFill="1"/></cellXfs></styleSheet>'
    )


def _xlsx_sheet_xml(rows: list[dict[str, object]]) -> str:
    headers = _sheet_headers(rows)
    matrix = [headers] + [[row.get(header) for header in headers] for row in rows]
    row_count = max(1, len(matrix))
    col_count = max(1, len(headers))
    dimension = f"A1:{_excel_column(col_count)}{row_count}"
    cols_xml = _xlsx_columns_xml(headers, rows)
    rows_xml = "\n".join(
        _xlsx_row_xml(row_index, row_values, header=row_index == 1)
        for row_index, row_values in enumerate(matrix, start=1)
    )
    auto_filter = f'<autoFilter ref="{dimension}"/>' if rows and headers else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"{cols_xml}<sheetData>{rows_xml}</sheetData>{auto_filter}</worksheet>"
    )


def _sheet_headers(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["empty"]
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    return headers


def _xlsx_columns_xml(headers: Sequence[str], rows: Sequence[dict[str, object]]) -> str:
    columns = []
    for index, header in enumerate(headers, start=1):
        width = min(
            42,
            max(
                10,
                len(str(header)) + 2,
                *[len(_xlsx_display_value(row.get(header))) + 2 for row in rows[:200]],
            ),
        )
        columns.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
    return f"<cols>{''.join(columns)}</cols>" if columns else ""


def _xlsx_row_xml(row_index: int, values: Sequence[object], *, header: bool = False) -> str:
    cells = [
        _xlsx_cell_xml(row_index, col_index, value, style=1 if header else 0)
        for col_index, value in enumerate(values, start=1)
    ]
    return f'<row r="{row_index}">{"".join(cells)}</row>'


def _xlsx_cell_xml(row_index: int, col_index: int, value: object, *, style: int = 0) -> str:
    reference = f"{_excel_column(col_index)}{row_index}"
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{reference}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return f'<c r="{reference}"{style_attr}><v>{float(value):.12g}</v></c>'
    text = escape(_xlsx_display_value(value))
    return f'<c r="{reference}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def _xlsx_display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _excel_column(index: int) -> str:
    letters = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters)) or "A"


def _format_optional(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.1f} m"


def _with_safe_constant_altitude(
    config: LocalizationConfig,
    terrain: TerrainManager,
) -> LocalizationConfig:
    if not config.terrain.dem_path:
        return config
    minimum_safe_msl = float(
        math.ceil((terrain.get_source_max_elevation() + config.sensor.min_safe_agl_m) / 10.0)
        * 10.0
    )
    if config.sensor.constant_msl_m >= minimum_safe_msl:
        return config
    return dataclass_replace(
        config,
        sensor=dataclass_replace(config.sensor, constant_msl_m=minimum_safe_msl),
    )


def _benchmark_algorithm(config: LocalizationConfig, dem_shape: tuple[int, int]):
    algorithm = config.algorithm
    max_edge = max(dem_shape)
    coarse_stride = algorithm.coarse_stride
    if max_edge > 1024:
        coarse_stride = max(coarse_stride, 32)
    elif max_edge > 512:
        coarse_stride = max(coarse_stride, 20)
    top_k = max(3, min(algorithm.top_k, 5))
    return dataclass_replace(algorithm, coarse_stride=coarse_stride, top_k=top_k)


def _simulate_route_samples(
    config: LocalizationConfig,
    terrain: TerrainManager,
    route: BenchmarkRoute,
    *,
    dense_sample_count: int,
    seed_offset: int,
) -> list[BenchmarkSample]:
    sensor = SensorSimulator(config.sensor, seed=config.terrain.seed + 1000 + seed_offset)
    distances = np.linspace(0.0, route.total_distance_m, dense_sample_count)
    previous_distance = 0.0
    samples: list[BenchmarkSample] = []

    for distance_m in distances:
        x, y, heading = _point_on_route(route, float(distance_m))
        traveled = float(distance_m - previous_distance)
        dt_s = traveled / max(config.route.speed_m_s, 1e-9)
        terrain_elevation = terrain.sample_elevation_at_world(x, y)
        measurement = sensor.generate_measurement(
            true_msl_m=config.sensor.constant_msl_m,
            terrain_elevation_m=terrain_elevation,
            true_heading_deg=heading,
            traveled_distance_m=traveled,
            dt_s=dt_s,
        )
        samples.append(
            BenchmarkSample(
                distance_m=float(distance_m),
                x=x,
                y=y,
                heading_deg=heading,
                measurement=measurement,
            )
        )
        previous_distance = float(distance_m)

    return samples


def _samples_until_fraction(
    samples: Sequence[BenchmarkSample],
    checkpoint_fraction: float,
) -> list[BenchmarkSample]:
    if not samples:
        return []
    target_distance_m = samples[-1].distance_m * max(0.01, min(1.0, float(checkpoint_fraction)))
    distances = np.asarray([sample.distance_m for sample in samples], dtype=np.float64)
    end_index = int(np.searchsorted(distances, target_distance_m, side="right"))
    end_index = min(len(samples), max(2, end_index))
    return list(samples[:end_index])


def _point_on_route(route: BenchmarkRoute, distance_m: float) -> tuple[float, float, float]:
    cursor_x = route.start_x
    cursor_y = route.start_y
    remaining = max(0.0, min(distance_m, route.total_distance_m))
    last_heading = route.segments[-1][0] if route.segments else 0.0
    for heading_deg, segment_distance in route.segments:
        if remaining <= segment_distance:
            dx, dy = _offset_meters(remaining, heading_deg)
            return cursor_x + dx, cursor_y + dy, float(heading_deg)
        dx, dy = _offset_meters(segment_distance, heading_deg)
        cursor_x += dx
        cursor_y += dy
        remaining -= segment_distance
        last_heading = heading_deg
    return cursor_x, cursor_y, float(last_heading)


def _offset_meters(distance_m: float, heading_deg: float) -> tuple[float, float]:
    angle_rad = math.radians(normalize_heading(heading_deg))
    return distance_m * math.sin(angle_rad), distance_m * math.cos(angle_rad)


def _profile_for_variant(
    samples: Sequence[BenchmarkSample],
    variant: BenchmarkVariant,
) -> BenchmarkProfile:
    if variant.mode == "raw":
        selected = list(samples[-variant.point_count :])
        return _raw_profile(selected)
    if variant.mode == "interp":
        return _interpolated_profile(samples, variant.point_count)
    raise ValueError(f"Unknown benchmark vector mode: {variant.mode}")


def _raw_profile(samples: Sequence[BenchmarkSample]) -> BenchmarkProfile:
    first = samples[0]
    distances = np.asarray([sample.distance_m - first.distance_m for sample in samples])
    offsets = [
        (
            float(sample.x - first.x),
            float(sample.y - first.y),
            sample.measurement.sensor_heading_deg,
        )
        for sample in samples
    ]
    return BenchmarkProfile(
        distances_m=distances,
        offsets=offsets,
        laser_agl=np.asarray([sample.measurement.laser_agl_m for sample in samples]),
        laser_valid=np.asarray([sample.measurement.laser_valid for sample in samples], dtype=bool),
        baro_msl=np.asarray([sample.measurement.baro_msl_m for sample in samples]),
        true_current_x=samples[-1].x,
        true_current_y=samples[-1].y,
        true_current_heading_deg=samples[-1].heading_deg,
    )


def _interpolated_profile(
    samples: Sequence[BenchmarkSample],
    point_count: int,
) -> BenchmarkProfile:
    count = max(2, int(point_count))
    source_distances = np.asarray([sample.distance_m for sample in samples], dtype=np.float64)
    target_distances = np.linspace(source_distances[0], source_distances[-1], count)
    xs = np.interp(target_distances, source_distances, [sample.x for sample in samples])
    ys = np.interp(target_distances, source_distances, [sample.y for sample in samples])
    baro = np.interp(
        target_distances,
        source_distances,
        [sample.measurement.baro_msl_m for sample in samples],
    )

    source_valid = np.asarray([sample.measurement.laser_valid for sample in samples], dtype=bool)
    source_laser = np.asarray([sample.measurement.laser_agl_m for sample in samples])
    valid_distances = source_distances[source_valid]
    if len(valid_distances) >= 2:
        laser = np.interp(target_distances, valid_distances, source_laser[source_valid])
        source_step = float(np.median(np.diff(source_distances))) if len(source_distances) > 1 else 0.0
        max_gap = max(source_step * 2.5, 1e-9)
        nearest_right = np.searchsorted(valid_distances, target_distances, side="left")
        nearest_left = np.maximum(nearest_right - 1, 0)
        nearest_right = np.minimum(nearest_right, len(valid_distances) - 1)
        nearest_gap = np.minimum(
            np.abs(target_distances - valid_distances[nearest_left]),
            np.abs(target_distances - valid_distances[nearest_right]),
        )
        laser_valid = nearest_gap <= max_gap
    else:
        laser = np.full(count, np.nan, dtype=np.float64)
        laser_valid = np.zeros(count, dtype=bool)

    heading_indices = np.searchsorted(source_distances, target_distances, side="right") - 1
    heading_indices = np.clip(heading_indices, 0, len(samples) - 1)
    headings = [
        samples[int(index)].measurement.sensor_heading_deg for index in heading_indices.tolist()
    ]
    first_x = float(xs[0])
    first_y = float(ys[0])
    offsets = [
        (float(x - first_x), float(y - first_y), float(heading))
        for x, y, heading in zip(xs, ys, headings, strict=True)
    ]
    return BenchmarkProfile(
        distances_m=target_distances - target_distances[0],
        offsets=offsets,
        laser_agl=laser,
        laser_valid=laser_valid,
        baro_msl=baro,
        true_current_x=float(xs[-1]),
        true_current_y=float(ys[-1]),
        true_current_heading_deg=samples[-1].heading_deg,
    )


def _evaluate_profile(
    config: LocalizationConfig,
    matcher: ProfileMatcher,
    ct: CoordinateTransform,
    route: BenchmarkRoute,
    checkpoint_fraction: float,
    variant: BenchmarkVariant,
    profile: BenchmarkProfile,
    *,
    wrong_fix_threshold_m: float,
) -> tuple[BenchmarkSummary, ProfileComparison]:
    started = time.perf_counter()
    search_headings = [profile.offsets[0][2]] if profile.offsets else [0.0]
    candidates = matcher.coarse_to_fine_search(
        profile.laser_agl,
        profile.laser_valid,
        profile.baro_msl,
        profile.offsets,
        search_headings,
        search_bounds=None,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    candidate = candidates[0] if candidates else None
    accepted = candidate is not None and _candidate_passes_quality(config, candidate)
    error_m = None
    if candidate is not None:
        estimated_x, estimated_y, _heading = _estimate_current_state(candidate, profile, ct)
        error_m = math.hypot(
            estimated_x - profile.true_current_x,
            estimated_y - profile.true_current_y,
        )
    wrong_fix = (
        accepted
        and error_m is not None
        and error_m > max(0.0, float(wrong_fix_threshold_m))
    )

    metrics = candidate.metrics if candidate is not None else {}
    summary = BenchmarkSummary(
        route_name=route.name,
        checkpoint_fraction=float(checkpoint_fraction),
        checkpoint_distance_m=route.total_distance_m * float(checkpoint_fraction),
        route_total_distance_m=route.total_distance_m,
        variant_name=variant.name,
        vector_mode=variant.mode,
        requested_points=variant.point_count,
        used_points=len(profile.offsets),
        profile_span_m=float(profile.distances_m[-1] - profile.distances_m[0])
        if len(profile.distances_m) > 1
        else 0.0,
        fix_accepted=accepted,
        wrong_fix=wrong_fix,
        candidate_found=candidate is not None,
        position_error_m=error_m,
        score=candidate.score if candidate is not None else None,
        inlier_rmse_m=_metric_value(metrics, "inlier_rmse"),
        correlation=_metric_value(metrics, "inlier_correlation", "correlation"),
        valid_ratio=candidate.valid_ratio if candidate is not None else None,
        elapsed_ms=elapsed_ms,
    )
    status = "fix" if accepted else "quality_rejected" if candidate is not None else "no_candidates"
    return summary, _build_profile_comparison(config, matcher, profile, candidate, status)


def _candidate_passes_quality(config: LocalizationConfig, candidate: Candidate) -> bool:
    metrics = candidate.metrics
    quality_score = float(metrics.get("inlier_rmse", metrics.get("rmse", candidate.score)))
    correlation = float(metrics.get("inlier_correlation", metrics.get("correlation", 1.0)))
    return (
        quality_score <= config.algorithm.max_match_inlier_rmse_m
        and correlation >= config.algorithm.min_match_inlier_correlation
        and candidate.valid_ratio >= config.algorithm.min_match_valid_ratio
    )


def _estimate_current_state(
    candidate: Candidate,
    profile: BenchmarkProfile,
    ct: CoordinateTransform,
) -> tuple[float, float, float]:
    last_offset = profile.offsets[-1]
    angle_diff_deg = candidate.heading_deg - profile.offsets[0][2]
    rad = math.radians(angle_diff_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = last_offset[0]
    dy = last_offset[1]
    dx_new = dx * cos_a - dy * sin_a
    dy_new = dx * sin_a + dy * cos_a
    curr_row = candidate.row + (-dy_new / ct.dy)
    curr_col = candidate.col + (dx_new / ct.dx)
    curr_x, curr_y = ct.pixel_to_world(curr_row, curr_col)
    curr_h = normalize_heading(last_offset[2] + angle_diff_deg)
    return curr_x, curr_y, curr_h


def _build_profile_comparison(
    config: LocalizationConfig,
    matcher: ProfileMatcher,
    profile: BenchmarkProfile,
    candidate: Candidate | None,
    status: str,
) -> ProfileComparison:
    matched_dem = np.full(len(profile.laser_agl), np.nan, dtype=np.float64)
    if candidate is not None and profile.offsets:
        angle_delta = candidate.heading_deg - profile.offsets[0][2]
        matched_dem = extract_profile(
            matcher.dem,
            candidate.row,
            candidate.col,
            rotate_offsets(profile.offsets, angle_delta),
            matcher.ct,
        )

    measured = _measured_elevation(config, profile, matched_dem, candidate)
    quality_score = None
    quality_correlation = None
    if candidate is not None:
        quality_score = _metric_value(candidate.metrics, "inlier_rmse", "rmse")
        quality_correlation = _metric_value(
            candidate.metrics,
            "inlier_correlation",
            "correlation",
        )
    return ProfileComparison(
        distances_m=profile.distances_m.tolist(),
        measured_elevation_m=measured.tolist(),
        matched_elevation_m=matched_dem.tolist(),
        status=status,
        candidate_score=candidate.score if candidate is not None else None,
        quality_score=quality_score,
        quality_correlation=quality_correlation,
    )


def _measured_elevation(
    config: LocalizationConfig,
    profile: BenchmarkProfile,
    matched_dem: np.ndarray,
    candidate: Candidate | None,
) -> np.ndarray:
    mode = config.sensor.altitude_mode
    measured = np.full(len(profile.laser_agl), np.nan, dtype=np.float64)
    if mode == "known_msl_altitude":
        measured = float(config.sensor.constant_msl_m) - profile.laser_agl
    elif mode == "unknown_constant_msl_altitude":
        if candidate is not None and np.isfinite(candidate.estimated_msl_m):
            measured = float(candidate.estimated_msl_m) - profile.laser_agl
    elif mode == "barometric_altitude":
        valid_for_bias = profile.laser_valid & np.isfinite(matched_dem)
        if np.any(valid_for_bias):
            bias = float(
                np.median(
                    profile.laser_agl[valid_for_bias]
                    + matched_dem[valid_for_bias]
                    - profile.baro_msl[valid_for_bias]
                )
            )
            measured = profile.baro_msl + bias - profile.laser_agl
        else:
            measured = profile.baro_msl - profile.laser_agl
    else:
        raise ValueError(f"Unknown altitude mode: {mode}")

    measured = np.asarray(measured, dtype=np.float64)
    measured[~profile.laser_valid] = np.nan
    return measured


def _metric_value(metrics: dict[str, float], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return float(value)
    return None


def _thin_path(points: Sequence[tuple[float, float]], max_points: int = 400) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return [(float(x), float(y)) for x, y in points]
    step = max(1, math.ceil(len(points) / max_points))
    thinned = [(float(x), float(y)) for x, y in points[::step]]
    if thinned[-1] != points[-1]:
        x, y = points[-1]
        thinned.append((float(x), float(y)))
    return thinned


def _should_stop(stop_requested: StopCallback | None) -> bool:
    return bool(stop_requested is not None and stop_requested())
