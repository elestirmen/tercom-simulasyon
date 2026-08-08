"""Parameter optimization benchmark for online TERCOM localization."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from terrain_nav.benchmark import (
    DEFAULT_WRONG_FIX_THRESHOLD_M,
    BenchmarkRoute,
    _with_safe_constant_altitude,
    _write_xlsx,
    build_benchmark_routes,
)
from terrain_nav.config import (
    MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
    LocalizationConfig,
    apply_realistic_noise_mode,
)
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.metrics import EstimatedState
from terrain_nav.simulation import MotionOutOfBoundsError, SimulationEngine
from terrain_nav.terrain import TerrainManager

ProgressCallback = Callable[[str], None]
StopCallback = Callable[[], bool]


PROFILE_MODES = ("uniform", "interpolated")
PROFILE_POINTS = (50, 100, 150, 250)
COARSE_STRIDES = (5, 10, 16)
MEDIUM_STRIDES = (2, 3)
FINE_STRIDE = 1
INLIER_CORRELATIONS = (0.95, 0.98, 0.99)
INLIER_RMSE_VALUES = (2.0, 2.5, 3.0)
VALID_RATIOS = (0.85, 0.90, 0.95)
TRIM_FRACTIONS = (0.03, 0.05, 0.10)
TOP_K_VALUES = (3, 5, 7)
ROI_SIZES_PX = (0, 128, 256, 384, 512)
SPEED_RANGES = ((5.0, 30.0), (8.0, 24.0), (8.0, 20.0), (10.0, 20.0))
SPEED_COARSE_STEPS = (2.5, 5.0)
SPEED_MEDIUM_STEPS = (0.5, 1.0)
SPEED_FINE_STEPS = (0.2, 0.5, 1.0)
SPEED_KEEP_VALUES = (1, 2, 3, 5)
MIN_PROFILE_DURATIONS = (30.0, 60.0, 90.0)
MAX_PROFILE_DURATIONS = (90.0, 120.0)
HEADING_SCENARIOS = ("known_heading", "noisy_heading_3deg", "noisy_heading_5deg")


@dataclass(frozen=True)
class OptimizerCandidate:
    config_id: str
    source: str
    profile_mode: str
    profile_points: int
    coarse_stride: int
    medium_stride: int
    fine_stride: int
    min_match_inlier_correlation: float
    max_match_inlier_rmse_m: float
    min_match_valid_ratio: float
    quality_trim_fraction: float
    top_k: int
    search_roi_size_px: int
    speed_min_m_s: float
    speed_max_m_s: float
    speed_search_coarse_step_m_s: float
    speed_search_medium_step_m_s: float
    speed_search_fine_step_m_s: float
    speed_search_keep_hypotheses: int
    min_profile_duration_s: float
    max_profile_duration_s: float


@dataclass(frozen=True)
class OptimizerSummary:
    config_id: str
    dataset: str
    stage: str
    source: str
    profile_mode: str
    profile_points: int
    coarse_stride: int
    medium_stride: int
    fine_stride: int
    min_match_inlier_correlation: float
    max_match_inlier_rmse_m: float
    min_match_valid_ratio: float
    quality_trim_fraction: float
    top_k: int
    search_roi_size_px: int
    speed_min_m_s: float
    speed_max_m_s: float
    speed_search_coarse_step_m_s: float
    speed_search_medium_step_m_s: float
    speed_search_fine_step_m_s: float
    speed_search_keep_hypotheses: int
    min_profile_duration_s: float
    max_profile_duration_s: float
    runs: int
    route_count: int
    heading_count: int
    total_updates: int
    correct_fix_count: int
    false_fix_count: int
    fix_count: int
    ambiguous_count: int
    no_fix_count: int
    correct_fix_rate: float
    false_fix_rate: float
    fix_precision: float
    fix_rate: float
    ambiguous_rate: float
    no_fix_rate: float
    median_position_error_m: float | None
    p95_position_error_m: float | None
    mean_position_error_m: float | None
    speed_mae_m_s: float | None
    speed_median_error_m_s: float | None
    speed_p95_error_m_s: float | None
    speed_ambiguity_rate: float
    initial_global_fix_time_ms: float | None
    initial_fix_timestamp_s: float | None
    mean_tracking_time_ms: float | None
    median_tracking_time_ms: float | None
    p95_tracking_time_ms: float | None
    mean_runtime_ms: float | None
    p95_runtime_ms: float | None
    coarse_speed_search_ms: float | None
    medium_speed_search_ms: float | None
    fine_speed_search_ms: float | None
    dem_profile_matching_ms: float | None
    quality_gate_ms: float | None
    position_ambiguity_ms: float | None
    speed_ambiguity_ms: float | None
    mean_speed_hypotheses: float | None
    mean_dem_searches: float | None
    mean_spatial_candidates: float | None
    mean_searched_dem_area: float | None
    recovery_count: int
    global_research_count: int
    score: float
    rank_key: str
    pareto_frontier: bool = False
    selection_role: str = ""


@dataclass(frozen=True)
class OptimizerResult:
    validation_summaries: list[OptimizerSummary]
    final_summaries: list[OptimizerSummary]
    raw_details: list[dict[str, object]]
    eliminated_rows: list[dict[str, object]]
    selected_config_ids: dict[str, str]
    route_split: dict[str, list[str]]
    total_candidate_count: int
    evaluated_candidate_count: int
    default_final: OptimizerSummary | None = None
    optimized_final: OptimizerSummary | None = None
    excel_path: str | None = None
    summary_csv_path: str | None = None
    details_jsonl_path: str | None = None


@dataclass(frozen=True)
class OptimizerRunConfig:
    initial_config_limit: int = 64
    refined_config_limit: int = 12
    final_config_limit: int = 10
    max_routes: int = 12
    stage1_route_limit: int = 4
    stage2_route_limit: int = 8
    final_route_limit: int = 4
    heading_scenarios: tuple[str, ...] = HEADING_SCENARIOS
    wrong_fix_threshold_m: float = DEFAULT_WRONG_FIX_THRESHOLD_M
    benchmark_sample_spacing_m: float | None = None
    max_updates_per_route: int = 0


def run_optimizer_benchmark(
    config: LocalizationConfig,
    *,
    run_config: OptimizerRunConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
    output_dir: str | Path | None = None,
) -> OptimizerResult:
    """Run a deterministic staged sweep and evaluate selected configs on unseen routes."""
    run_config = run_config or OptimizerRunConfig()
    terrain = TerrainManager(config.terrain)
    try:
        config = _prepare_base_config(config, terrain)
        ct = CoordinateTransform(terrain.dx, terrain.dy)
        routes = build_benchmark_routes(terrain, max_routes=run_config.max_routes)
        validation_routes, final_routes = split_optimizer_routes(routes)
        validation_routes = validation_routes[: max(1, run_config.stage2_route_limit)]
        final_routes = final_routes[: max(1, run_config.final_route_limit)]
        stage1_routes = validation_routes[: max(1, run_config.stage1_route_limit)]
        candidates = generate_optimizer_candidates(
            config,
            limit=max(1, run_config.initial_config_limit),
        )
        if progress_callback is not None:
            progress_callback(
                "[OPT] Plan: "
                f"{len(candidates)} aday, {len(stage1_routes)} coarse rota, "
                f"{len(validation_routes)} validation rota, {len(final_routes)} final rota"
            )

        validation_summaries: list[OptimizerSummary] = []
        raw_details: list[dict[str, object]] = []
        eliminated_rows: list[dict[str, object]] = []

        stage1_summaries = _evaluate_candidate_set(
            config,
            ct,
            candidates,
            stage1_routes,
            dataset="validation",
            stage="coarse_sweep",
            heading_scenarios=run_config.heading_scenarios,
            wrong_fix_threshold_m=run_config.wrong_fix_threshold_m,
            benchmark_sample_spacing_m=run_config.benchmark_sample_spacing_m,
            max_updates_per_route=run_config.max_updates_per_route,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )
        validation_summaries.extend(stage1_summaries[0])
        raw_details.extend(stage1_summaries[1])
        refined_candidates = _select_ranked_candidates(
            validation_summaries,
            candidates,
            limit=max(1, run_config.refined_config_limit),
        )
        eliminated_rows.extend(
            _eliminated_rows(
                candidates,
                refined_candidates,
                "coarse_sweep_rank",
            )
        )

        if not _should_stop(stop_requested):
            refined = _evaluate_candidate_set(
                config,
                ct,
                refined_candidates,
                validation_routes,
                dataset="validation",
                stage="refined_validation",
                heading_scenarios=run_config.heading_scenarios,
                wrong_fix_threshold_m=run_config.wrong_fix_threshold_m,
                benchmark_sample_spacing_m=run_config.benchmark_sample_spacing_m,
                max_updates_per_route=run_config.max_updates_per_route,
                progress_callback=progress_callback,
                stop_requested=stop_requested,
            )
            validation_summaries.extend(refined[0])
            raw_details.extend(refined[1])

        refined_summaries = [
            summary
            for summary in validation_summaries
            if summary.stage == "refined_validation"
        ] or validation_summaries
        pareto_ids = {
            summary.config_id
            for summary in _pareto_frontier(refined_summaries)
        }
        final_candidates = _select_final_candidates(
            refined_summaries,
            refined_candidates,
            pareto_ids=pareto_ids,
            limit=max(1, run_config.final_config_limit),
        )
        eliminated_rows.extend(
            _eliminated_rows(
                refined_candidates,
                final_candidates,
                "refined_validation_rank",
            )
        )

        final_summaries: list[OptimizerSummary] = []
        if not _should_stop(stop_requested):
            final_result = _evaluate_candidate_set(
                config,
                ct,
                final_candidates,
                final_routes,
                dataset="final_test",
                stage="unseen_final_test",
                heading_scenarios=run_config.heading_scenarios,
                wrong_fix_threshold_m=run_config.wrong_fix_threshold_m,
                benchmark_sample_spacing_m=run_config.benchmark_sample_spacing_m,
                max_updates_per_route=run_config.max_updates_per_route,
                progress_callback=progress_callback,
                stop_requested=stop_requested,
            )
            final_summaries.extend(final_result[0])
            raw_details.extend(final_result[1])

        selected = _selection_roles(refined_summaries)
        selected.update(_selection_roles(final_summaries))
        validation_summaries = _mark_summaries(validation_summaries, pareto_ids, selected)
        final_summaries = _mark_summaries(final_summaries, pareto_ids, selected)

        default_candidate = default_unknown_speed_candidate(config)
        default_final = None
        if final_routes and not _should_stop(stop_requested):
            default_result = _evaluate_candidate_set(
                config,
                ct,
                [default_candidate],
                final_routes,
                dataset="final_test",
                stage="default_unknown_speed",
                heading_scenarios=run_config.heading_scenarios,
                wrong_fix_threshold_m=run_config.wrong_fix_threshold_m,
                benchmark_sample_spacing_m=run_config.benchmark_sample_spacing_m,
                max_updates_per_route=run_config.max_updates_per_route,
                progress_callback=progress_callback,
                stop_requested=stop_requested,
            )
            default_final = default_result[0][0] if default_result[0] else None
            raw_details.extend(default_result[1])

        optimized_final = _summary_by_id(final_summaries, selected.get("balanced"))
        result = OptimizerResult(
            validation_summaries=validation_summaries,
            final_summaries=final_summaries,
            raw_details=raw_details,
            eliminated_rows=eliminated_rows,
            selected_config_ids=selected,
            route_split={
                "validation": [route.name for route in validation_routes],
                "final_test": [route.name for route in final_routes],
            },
            total_candidate_count=len(candidates),
            evaluated_candidate_count=len({summary.config_id for summary in validation_summaries}),
            default_final=default_final,
            optimized_final=optimized_final,
        )
        if output_dir is not None:
            result = save_optimizer_result(result, output_dir)
        return result
    finally:
        terrain.close()


def generate_optimizer_candidates(
    config: LocalizationConfig,
    *,
    limit: int,
) -> list[OptimizerCandidate]:
    """Build a deterministic staged candidate set without materializing the full grid."""
    base = _balanced_candidate(config)
    candidates: list[OptimizerCandidate] = [base]
    values: dict[str, Sequence[object]] = {
        "profile_mode": PROFILE_MODES,
        "profile_points": PROFILE_POINTS,
        "min_profile_duration_s": MIN_PROFILE_DURATIONS,
        "max_profile_duration_s": MAX_PROFILE_DURATIONS,
        "min_match_inlier_correlation": INLIER_CORRELATIONS,
        "max_match_inlier_rmse_m": INLIER_RMSE_VALUES,
        "min_match_valid_ratio": VALID_RATIOS,
        "quality_trim_fraction": TRIM_FRACTIONS,
        "coarse_stride": COARSE_STRIDES,
        "medium_stride": MEDIUM_STRIDES,
        "top_k": TOP_K_VALUES,
        "search_roi_size_px": ROI_SIZES_PX,
        "speed_range": SPEED_RANGES,
        "speed_search_coarse_step_m_s": SPEED_COARSE_STEPS,
        "speed_search_medium_step_m_s": SPEED_MEDIUM_STEPS,
        "speed_search_fine_step_m_s": SPEED_FINE_STEPS,
        "speed_search_keep_hypotheses": SPEED_KEEP_VALUES,
    }

    for field_name, field_values in values.items():
        for value in field_values:
            candidates.append(_candidate_with_value(base, field_name, value, "one_factor"))

    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
    field_names = list(values)
    index = 0
    while len(_unique_candidates(candidates)) < limit and index < limit * 8:
        kwargs = asdict(base)
        kwargs["source"] = "balanced_sample"
        for field_index, field_name in enumerate(field_names):
            field_values = values[field_name]
            value = field_values[(index * primes[field_index] + field_index) % len(field_values)]
            if field_name == "speed_range":
                kwargs["speed_min_m_s"], kwargs["speed_max_m_s"] = value
            else:
                kwargs[field_name] = value
        if kwargs["max_profile_duration_s"] <= kwargs["min_profile_duration_s"]:
            kwargs["max_profile_duration_s"] = 120.0
        candidates.append(OptimizerCandidate(**kwargs))
        index += 1

    unique = _unique_candidates(candidates)
    return [
        replace(candidate, config_id=f"opt_{index:04d}")
        for index, candidate in enumerate(unique[:limit], start=1)
    ]


def default_unknown_speed_candidate(config: LocalizationConfig) -> OptimizerCandidate:
    algorithm = config.algorithm
    return OptimizerCandidate(
        config_id="default_unknown_speed",
        source="current_default",
        profile_mode=algorithm.profile_resampling_mode,
        profile_points=algorithm.profile_points,
        coarse_stride=algorithm.coarse_stride,
        medium_stride=algorithm.medium_stride,
        fine_stride=algorithm.fine_stride,
        min_match_inlier_correlation=algorithm.min_match_inlier_correlation,
        max_match_inlier_rmse_m=algorithm.max_match_inlier_rmse_m,
        min_match_valid_ratio=algorithm.min_match_valid_ratio,
        quality_trim_fraction=algorithm.quality_trim_fraction,
        top_k=algorithm.top_k,
        search_roi_size_px=algorithm.search_roi_size_px,
        speed_min_m_s=algorithm.speed_search_min_m_s,
        speed_max_m_s=algorithm.speed_search_max_m_s,
        speed_search_coarse_step_m_s=algorithm.speed_search_coarse_step_m_s,
        speed_search_medium_step_m_s=algorithm.speed_search_medium_step_m_s,
        speed_search_fine_step_m_s=algorithm.speed_search_fine_step_m_s,
        speed_search_keep_hypotheses=algorithm.speed_search_keep_hypotheses,
        min_profile_duration_s=algorithm.min_profile_duration_s,
        max_profile_duration_s=algorithm.max_profile_duration_s,
    )


def split_optimizer_routes(
    routes: Sequence[BenchmarkRoute],
) -> tuple[list[BenchmarkRoute], list[BenchmarkRoute]]:
    final_routes = [route for index, route in enumerate(routes) if index % 4 == 3]
    validation_routes = [route for index, route in enumerate(routes) if index % 4 != 3]
    if routes and not final_routes:
        final_routes = [routes[-1]]
        validation_routes = list(routes[:-1]) or [routes[0]]
    if not validation_routes:
        validation_routes = list(routes)
    return validation_routes, final_routes


def save_optimizer_result(result: OptimizerResult, output_dir: str | Path) -> OptimizerResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"optimizer_{stamp}_summary.csv"
    details_path = out_dir / f"optimizer_{stamp}_details.jsonl"
    excel_path = out_dir / f"optimizer_{stamp}.xlsx"

    rows = [asdict(summary) for summary in [*result.validation_summaries, *result.final_summaries]]
    fieldnames = list(rows[0]) if rows else list(OptimizerSummary.__dataclass_fields__)
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with details_path.open("w", encoding="utf-8") as jsonl_file:
        for detail in result.raw_details:
            jsonl_file.write(json.dumps(detail, ensure_ascii=False) + "\n")

    saved = OptimizerResult(
        validation_summaries=result.validation_summaries,
        final_summaries=result.final_summaries,
        raw_details=result.raw_details,
        eliminated_rows=result.eliminated_rows,
        selected_config_ids=result.selected_config_ids,
        route_split=result.route_split,
        total_candidate_count=result.total_candidate_count,
        evaluated_candidate_count=result.evaluated_candidate_count,
        default_final=result.default_final,
        optimized_final=result.optimized_final,
        excel_path=str(excel_path),
        summary_csv_path=str(summary_path),
        details_jsonl_path=str(details_path),
    )
    write_optimizer_excel(saved, excel_path)
    return saved


def write_optimizer_excel(result: OptimizerResult, path: str | Path) -> None:
    validation_rows = [asdict(summary) for summary in result.validation_summaries]
    final_rows = [asdict(summary) for summary in result.final_summaries]
    sheets = [
        ("Genel Ozet", optimizer_overview_rows(result)),
        ("Top Configurations", optimizer_top_configuration_rows(result, limit=10)),
        ("Pareto Frontier", [asdict(row) for row in _pareto_frontier(result.validation_summaries)]),
        ("Final Test Results", final_rows),
        ("Quality Gate Analysis", _analysis_rows(result.validation_summaries, ("min_match_inlier_correlation", "max_match_inlier_rmse_m", "min_match_valid_ratio", "quality_trim_fraction"))),
        ("Profile Analysis", _analysis_rows(result.validation_summaries, ("profile_mode", "profile_points"))),
        ("ROI Analysis", _analysis_rows(result.validation_summaries, ("search_roi_size_px",))),
        ("Speed Search Analysis", _analysis_rows(result.validation_summaries, ("speed_min_m_s", "speed_max_m_s", "speed_search_coarse_step_m_s", "speed_search_medium_step_m_s", "speed_search_fine_step_m_s", "speed_search_keep_hypotheses"))),
        ("Profile Duration Analysis", _analysis_rows(result.validation_summaries, ("min_profile_duration_s", "max_profile_duration_s"))),
        ("Heading Analysis", _heading_analysis_rows(result.raw_details)),
        ("Runtime Breakdown", _runtime_breakdown_rows(result.validation_summaries)),
        ("Validation Results", validation_rows),
        ("Raw Details", result.raw_details),
        ("Eliminated", result.eliminated_rows),
    ]
    _write_xlsx(Path(path), sheets)


def optimizer_overview_rows(result: OptimizerResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"metric": "Aday konfigurasyon", "value": result.total_candidate_count},
        {"metric": "Degerlendirilen konfigurasyon", "value": result.evaluated_candidate_count},
        {"metric": "Validation rota", "value": len(result.route_split.get("validation", []))},
        {"metric": "Final test rota", "value": len(result.route_split.get("final_test", []))},
    ]
    for role, config_id in result.selected_config_ids.items():
        rows.append({"metric": f"Secim: {role}", "value": config_id})
    if result.default_final is not None and result.optimized_final is not None:
        rows.extend(_default_vs_optimized_rows(result.default_final, result.optimized_final))
    if result.excel_path:
        rows.append({"metric": "Excel dosyasi", "value": result.excel_path})
    if result.summary_csv_path:
        rows.append({"metric": "CSV dosyasi", "value": result.summary_csv_path})
    if result.details_jsonl_path:
        rows.append({"metric": "JSONL dosyasi", "value": result.details_jsonl_path})
    return rows


def optimizer_top_configuration_rows(
    result: OptimizerResult,
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    summaries = _latest_validation_summaries(result.validation_summaries)
    rows = [asdict(summary) for summary in sorted(summaries, key=_summary_sort_key)[:limit]]
    return rows


def optimizer_final_rows(result: OptimizerResult) -> list[dict[str, object]]:
    return [asdict(summary) for summary in sorted(result.final_summaries, key=_summary_sort_key)]


def format_optimizer_summary(result: OptimizerResult, *, limit: int = 10) -> str:
    if not result.validation_summaries:
        return "Optimizer sonucu yok."
    top = optimizer_top_configuration_rows(result, limit=limit)
    lines = [
        "Optimizer Benchmark: "
        f"{result.total_candidate_count} aday, "
        f"{len(result.route_split.get('validation', []))} validation rota, "
        f"{len(result.route_split.get('final_test', []))} final rota"
    ]
    balanced = result.optimized_final or _summary_by_id(
        result.validation_summaries,
        result.selected_config_ids.get("balanced"),
    )
    if balanced is not None:
        lines.append(
            "Onerilen: "
            f"{balanced.config_id} | false {balanced.false_fix_rate:.1%}, "
            f"correct {balanced.correct_fix_rate:.1%}, precision {balanced.fix_precision:.1%}, "
            f"P95 {_format_optional(balanced.p95_position_error_m)}, "
            f"tracking {_format_optional_ms(balanced.p95_tracking_time_ms)}"
        )
    for row in top[:limit]:
        lines.append(
            f"{row['config_id']}: false {row['false_fix_rate']:.1%}, "
            f"correct {row['correct_fix_rate']:.1%}, precision {row['fix_precision']:.1%}, "
            f"P95 {_format_optional(row['p95_position_error_m'])}, "
            f"score {row['score']:.3f}"
        )
    if result.excel_path:
        lines.append(f"Excel: {result.excel_path}")
    return "\n".join(lines)


def _evaluate_candidate_set(
    base_config: LocalizationConfig,
    ct: CoordinateTransform,
    candidates: Sequence[OptimizerCandidate],
    routes: Sequence[BenchmarkRoute],
    *,
    dataset: str,
    stage: str,
    heading_scenarios: Sequence[str],
    wrong_fix_threshold_m: float,
    benchmark_sample_spacing_m: float | None,
    max_updates_per_route: int,
    progress_callback: ProgressCallback | None,
    stop_requested: StopCallback | None,
) -> tuple[list[OptimizerSummary], list[dict[str, object]]]:
    summaries = []
    details = []
    total_jobs = len(candidates) * len(routes) * len(heading_scenarios)
    completed = 0
    for candidate in candidates:
        if _should_stop(stop_requested):
            break
        candidate_details = []
        for route_index, route in enumerate(routes):
            if _should_stop(stop_requested):
                break
            for heading_scenario in heading_scenarios:
                if _should_stop(stop_requested):
                    break
                run_details = _run_online_route(
                    base_config,
                    ct,
                    candidate,
                    route,
                    route_index=route_index,
                    dataset=dataset,
                    stage=stage,
                    heading_scenario=heading_scenario,
                    wrong_fix_threshold_m=wrong_fix_threshold_m,
                    benchmark_sample_spacing_m=benchmark_sample_spacing_m,
                    max_updates_per_route=max_updates_per_route,
                )
                candidate_details.extend(run_details)
                details.extend(run_details)
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        f"[OPT {completed}/{total_jobs}] {stage} {candidate.config_id} "
                        f"{route.name} {heading_scenario}"
                    )
        summaries.append(
            _aggregate_candidate(candidate, candidate_details, dataset=dataset, stage=stage)
        )
    return summaries, details


def _run_online_route(
    base_config: LocalizationConfig,
    ct: CoordinateTransform,
    candidate: OptimizerCandidate,
    route: BenchmarkRoute,
    *,
    route_index: int,
    dataset: str,
    stage: str,
    heading_scenario: str,
    wrong_fix_threshold_m: float,
    benchmark_sample_spacing_m: float | None,
    max_updates_per_route: int,
) -> list[dict[str, object]]:
    config = _config_for_candidate(
        base_config,
        ct,
        candidate,
        route,
        heading_scenario,
        benchmark_sample_spacing_m=benchmark_sample_spacing_m,
    )
    simulation = SimulationEngine(config, manual_control=True)
    details: list[dict[str, object]] = []
    first_correct_fix_seen = False
    update_count = 0
    try:
        for segment_heading, segment_distance in route.segments:
            turn = ((segment_heading - simulation.dynamic_h + 180.0) % 360.0) - 180.0
            if abs(turn) > 1e-9:
                simulation.turn_vehicle(turn)
            remaining = float(segment_distance)
            spacing = max(1e-9, float(config.route.manual_sample_spacing_m))
            while remaining > 1e-9:
                step_distance = min(spacing, remaining)
                phase_before = "tracking" if first_correct_fix_seen else "initial_global"
                try:
                    true_state, estimate, measurement = simulation.execute_motion(step_distance, 0.0)
                except MotionOutOfBoundsError:
                    return details
                runtime = simulation.get_runtime_profile()
                search_status = simulation.get_localization_status()
                row = _step_detail_row(
                    candidate,
                    route,
                    route_index,
                    dataset,
                    stage,
                    heading_scenario,
                    config,
                    true_state,
                    estimate,
                    measurement_timestamp=measurement.timestamp_s,
                    runtime=runtime,
                    search_status=search_status,
                    phase_before=phase_before,
                    wrong_fix_threshold_m=wrong_fix_threshold_m,
                )
                details.append(row)
                if row["correct_fix"]:
                    first_correct_fix_seen = True
                if (
                    config.motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED
                    and (measurement.traveled_distance_m is not None or measurement.measured_speed_m_s is not None)
                ):
                    row["ground_truth_leak_detected"] = True
                update_count += 1
                if max_updates_per_route > 0 and update_count >= max_updates_per_route:
                    return details
                remaining -= step_distance
        return details
    finally:
        simulation.close()


def _step_detail_row(
    candidate: OptimizerCandidate,
    route: BenchmarkRoute,
    route_index: int,
    dataset: str,
    stage: str,
    heading_scenario: str,
    config: LocalizationConfig,
    true_state: tuple[float, float, float],
    estimate: EstimatedState | None,
    *,
    measurement_timestamp: float,
    runtime: dict[str, float],
    search_status: dict,
    phase_before: str,
    wrong_fix_threshold_m: float,
) -> dict[str, object]:
    position_error = None
    speed_error = None
    ambiguous = False
    speed_ambiguous = False
    fix_accepted = False
    wrong_fix = False
    correct_fix = False
    if estimate is not None:
        ambiguous = bool(estimate.is_ambiguous)
        speed_ambiguous = bool(estimate.speed_is_ambiguous)
        fix_accepted = not ambiguous
        position_error = math.hypot(
            estimate.estimated_x - true_state[0],
            estimate.estimated_y - true_state[1],
        )
        if estimate.estimated_speed_m_s is not None:
            speed_error = abs(estimate.estimated_speed_m_s - config.route.speed_m_s)
        wrong_fix = bool(
            fix_accepted
            and position_error is not None
            and position_error > wrong_fix_threshold_m
        )
        correct_fix = bool(fix_accepted and not wrong_fix)

    return {
        **asdict(candidate),
        "dataset": dataset,
        "stage": stage,
        "route_name": route.name,
        "route_index": route_index,
        "heading_scenario": heading_scenario,
        "sensor_model": "realistic",
        "benchmark_sample_spacing_m": float(config.route.manual_sample_spacing_m),
        "true_speed_m_s": float(config.route.speed_m_s),
        "timestamp_s": float(measurement_timestamp),
        "phase": phase_before,
        "search_mode": search_status.get("mode"),
        "search_phase": search_status.get("phase"),
        "fix_accepted": fix_accepted,
        "correct_fix": correct_fix,
        "wrong_fix": wrong_fix,
        "ambiguous": ambiguous,
        "no_fix": estimate is None,
        "position_error_m": position_error,
        "speed_error_m_s": speed_error,
        "speed_ambiguous": speed_ambiguous,
        "recovery_active": search_status.get("phase") == "recovery",
        "global_research": (
            search_status.get("mode") == "global_search"
            and search_status.get("phase") == "recovery"
        ),
        "ground_truth_leak_detected": False,
        **runtime,
    }


def _aggregate_candidate(
    candidate: OptimizerCandidate,
    details: Sequence[dict[str, object]],
    *,
    dataset: str,
    stage: str,
) -> OptimizerSummary:
    total = len(details)
    fix_count = sum(1 for row in details if row.get("fix_accepted"))
    false_count = sum(1 for row in details if row.get("wrong_fix"))
    correct_count = sum(1 for row in details if row.get("correct_fix"))
    ambiguous_count = sum(1 for row in details if row.get("ambiguous"))
    no_fix_count = sum(1 for row in details if row.get("no_fix"))
    position_errors = _numeric_values(row.get("position_error_m") for row in details if row.get("fix_accepted"))
    speed_errors = _numeric_values(row.get("speed_error_m_s") for row in details if row.get("fix_accepted"))
    runtimes = _numeric_values(row.get("total_localization_ms") for row in details)
    tracking_runtimes = _numeric_values(
        row.get("total_localization_ms") for row in details if row.get("phase") == "tracking"
    )
    initial_fixes = [
        row
        for row in details
        if row.get("phase") == "initial_global" and row.get("correct_fix")
    ]
    initial_fix_runtime = _min_value(row.get("total_localization_ms") for row in initial_fixes)
    initial_fix_timestamp = _min_value(row.get("timestamp_s") for row in initial_fixes)
    route_count = len({row.get("route_name") for row in details})
    heading_count = len({row.get("heading_scenario") for row in details})

    base_row = asdict(candidate)
    correct_rate = _rate(correct_count, total)
    false_rate = _rate(false_count, total)
    precision = _rate(correct_count, fix_count)
    ambiguous_rate = _rate(ambiguous_count, total)
    score = _combined_score(
        correct_rate=correct_rate,
        false_rate=false_rate,
        precision=precision,
        ambiguous_rate=ambiguous_rate,
        p95_error_m=_percentile(position_errors, 95),
        p95_tracking_ms=_percentile(tracking_runtimes, 95),
        speed_mae_m_s=_mean(speed_errors),
    )
    rank_key = json.dumps(_summary_rank_tuple_values(false_rate, correct_rate, precision, _percentile(position_errors, 95), _percentile(tracking_runtimes, 95), initial_fix_runtime, _mean(speed_errors)))

    return OptimizerSummary(
        **base_row,
        dataset=dataset,
        stage=stage,
        runs=route_count * heading_count,
        route_count=route_count,
        heading_count=heading_count,
        total_updates=total,
        correct_fix_count=correct_count,
        false_fix_count=false_count,
        fix_count=fix_count,
        ambiguous_count=ambiguous_count,
        no_fix_count=no_fix_count,
        correct_fix_rate=correct_rate,
        false_fix_rate=false_rate,
        fix_precision=precision,
        fix_rate=_rate(fix_count, total),
        ambiguous_rate=ambiguous_rate,
        no_fix_rate=_rate(no_fix_count, total),
        median_position_error_m=_median(position_errors),
        p95_position_error_m=_percentile(position_errors, 95),
        mean_position_error_m=_mean(position_errors),
        speed_mae_m_s=_mean(speed_errors),
        speed_median_error_m_s=_median(speed_errors),
        speed_p95_error_m_s=_percentile(speed_errors, 95),
        speed_ambiguity_rate=_rate(sum(1 for row in details if row.get("speed_ambiguous")), total),
        initial_global_fix_time_ms=initial_fix_runtime,
        initial_fix_timestamp_s=initial_fix_timestamp,
        mean_tracking_time_ms=_mean(tracking_runtimes),
        median_tracking_time_ms=_median(tracking_runtimes),
        p95_tracking_time_ms=_percentile(tracking_runtimes, 95),
        mean_runtime_ms=_mean(runtimes),
        p95_runtime_ms=_percentile(runtimes, 95),
        coarse_speed_search_ms=_mean(row.get("coarse_speed_search_ms") for row in details),
        medium_speed_search_ms=_mean(row.get("medium_speed_search_ms") for row in details),
        fine_speed_search_ms=_mean(row.get("fine_speed_search_ms") for row in details),
        dem_profile_matching_ms=_mean(row.get("dem_profile_matching_ms") for row in details),
        quality_gate_ms=_mean(row.get("quality_gate_ms") for row in details),
        position_ambiguity_ms=_mean(row.get("position_ambiguity_ms") for row in details),
        speed_ambiguity_ms=_mean(row.get("speed_ambiguity_ms") for row in details),
        mean_speed_hypotheses=_mean(row.get("speed_hypotheses_evaluated") for row in details),
        mean_dem_searches=_mean(row.get("dem_searches") for row in details),
        mean_spatial_candidates=_mean(row.get("spatial_candidates_evaluated") for row in details),
        mean_searched_dem_area=_mean(row.get("searched_dem_area_fraction_sum") for row in details),
        recovery_count=sum(1 for row in details if row.get("recovery_active")),
        global_research_count=sum(1 for row in details if row.get("global_research")),
        score=score,
        rank_key=rank_key,
    )


def _prepare_base_config(
    config: LocalizationConfig,
    terrain: TerrainManager,
) -> LocalizationConfig:
    prepared = replace(config, motion_mode=MOTION_MODE_UNKNOWN_CONSTANT_SPEED)
    prepared = _with_safe_constant_altitude(prepared, terrain)
    prepared = apply_realistic_noise_mode(
        prepared,
        True,
        fast_synthetic=not bool(prepared.terrain.dem_path),
    )
    return replace(prepared, motion_mode=MOTION_MODE_UNKNOWN_CONSTANT_SPEED)


def _config_for_candidate(
    base_config: LocalizationConfig,
    ct: CoordinateTransform,
    candidate: OptimizerCandidate,
    route: BenchmarkRoute,
    heading_scenario: str,
    benchmark_sample_spacing_m: float | None,
) -> LocalizationConfig:
    start_row, start_col = ct.world_to_pixel(route.start_x, route.start_y)
    algorithm = replace(
        base_config.algorithm,
        profile_resampling_mode=candidate.profile_mode,
        profile_points=int(candidate.profile_points),
        profile_window_size=max(250, int(candidate.profile_points), int(base_config.algorithm.profile_window_size)),
        min_profile_duration_s=float(candidate.min_profile_duration_s),
        max_profile_duration_s=float(candidate.max_profile_duration_s),
        coarse_stride=int(candidate.coarse_stride),
        medium_stride=int(candidate.medium_stride),
        fine_stride=int(candidate.fine_stride),
        min_match_inlier_correlation=float(candidate.min_match_inlier_correlation),
        max_match_inlier_rmse_m=float(candidate.max_match_inlier_rmse_m),
        min_match_valid_ratio=float(candidate.min_match_valid_ratio),
        quality_trim_fraction=float(candidate.quality_trim_fraction),
        top_k=int(candidate.top_k),
        search_roi_size_px=int(candidate.search_roi_size_px),
        speed_search_min_m_s=float(candidate.speed_min_m_s),
        speed_search_max_m_s=float(candidate.speed_max_m_s),
        speed_search_coarse_step_m_s=float(candidate.speed_search_coarse_step_m_s),
        speed_search_medium_step_m_s=float(candidate.speed_search_medium_step_m_s),
        speed_search_fine_step_m_s=float(candidate.speed_search_fine_step_m_s),
        speed_search_keep_hypotheses=int(candidate.speed_search_keep_hypotheses),
    )
    sensor = _sensor_for_heading(base_config, heading_scenario)
    route_config = replace(
        base_config.route,
        start_row=int(round(start_row)),
        start_col=int(round(start_col)),
        heading_deg=float(route.segments[0][0] if route.segments else base_config.route.heading_deg),
        route_length_m=float(route.total_distance_m),
        manual_step_distance_m=float(route.total_distance_m),
        manual_sample_spacing_m=float(
            benchmark_sample_spacing_m
            if benchmark_sample_spacing_m is not None
            else base_config.route.sample_spacing_m
        ),
    )
    return replace(
        base_config,
        sensor=sensor,
        algorithm=algorithm,
        route=route_config,
        motion_mode=MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
    )


def _sensor_for_heading(config: LocalizationConfig, heading_scenario: str):
    if heading_scenario == "known_heading":
        return replace(
            config.sensor,
            heading_mode="known_heading",
            sensor_heading_noise_std_deg=0.0,
            heading_uncertainty_deg=0.0,
        )
    if heading_scenario == "noisy_heading_3deg":
        return replace(
            config.sensor,
            heading_mode="noisy_heading",
            sensor_heading_noise_std_deg=1.0,
            heading_uncertainty_deg=3.0,
        )
    if heading_scenario == "noisy_heading_5deg":
        return replace(
            config.sensor,
            heading_mode="noisy_heading",
            sensor_heading_noise_std_deg=1.5,
            heading_uncertainty_deg=5.0,
        )
    raise ValueError(f"Unknown heading scenario: {heading_scenario}")


def _balanced_candidate(config: LocalizationConfig) -> OptimizerCandidate:
    algorithm = config.algorithm
    return OptimizerCandidate(
        config_id="opt_seed",
        source="balanced_seed",
        profile_mode="interpolated",
        profile_points=100,
        coarse_stride=10,
        medium_stride=3,
        fine_stride=1,
        min_match_inlier_correlation=0.98,
        max_match_inlier_rmse_m=3.0,
        min_match_valid_ratio=0.90,
        quality_trim_fraction=0.05,
        top_k=max(5, int(algorithm.top_k)),
        search_roi_size_px=256,
        speed_min_m_s=8.0,
        speed_max_m_s=24.0,
        speed_search_coarse_step_m_s=5.0,
        speed_search_medium_step_m_s=1.0,
        speed_search_fine_step_m_s=0.5,
        speed_search_keep_hypotheses=3,
        min_profile_duration_s=60.0,
        max_profile_duration_s=120.0,
    )


def _candidate_with_value(
    base: OptimizerCandidate,
    field_name: str,
    value: object,
    source: str,
) -> OptimizerCandidate:
    kwargs = asdict(base)
    kwargs["source"] = source
    if field_name == "speed_range":
        kwargs["speed_min_m_s"], kwargs["speed_max_m_s"] = value
    else:
        kwargs[field_name] = value
    if kwargs["max_profile_duration_s"] <= kwargs["min_profile_duration_s"]:
        kwargs["max_profile_duration_s"] = 120.0
    return OptimizerCandidate(**kwargs)


def _unique_candidates(candidates: Sequence[OptimizerCandidate]) -> list[OptimizerCandidate]:
    seen: set[tuple[object, ...]] = set()
    unique: list[OptimizerCandidate] = []
    for candidate in candidates:
        key = tuple(
            value
            for field, value in asdict(candidate).items()
            if field not in {"config_id", "source"}
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _select_ranked_candidates(
    summaries: Sequence[OptimizerSummary],
    candidates: Sequence[OptimizerCandidate],
    *,
    limit: int,
) -> list[OptimizerCandidate]:
    latest = _latest_validation_summaries(summaries)
    selected_ids: list[str] = []
    safety_limit = max(1, math.ceil(limit / 2))
    for summary in sorted(latest, key=_summary_sort_key)[:safety_limit]:
        if summary.config_id not in selected_ids:
            selected_ids.append(summary.config_id)
    for summary in sorted(latest, key=lambda row: row.score, reverse=True):
        if summary.config_id not in selected_ids:
            selected_ids.append(summary.config_id)
        if len(selected_ids) >= limit:
            break
    by_id = {candidate.config_id: candidate for candidate in candidates}
    return [by_id[config_id] for config_id in selected_ids if config_id in by_id]


def _select_final_candidates(
    summaries: Sequence[OptimizerSummary],
    candidates: Sequence[OptimizerCandidate],
    *,
    pareto_ids: set[str],
    limit: int,
) -> list[OptimizerCandidate]:
    roles = _selection_roles(summaries)
    ordered_ids = []
    for config_id in roles.values():
        if config_id and config_id not in ordered_ids:
            ordered_ids.append(config_id)
    for summary in sorted(summaries, key=_summary_sort_key):
        if summary.config_id in pareto_ids and summary.config_id not in ordered_ids:
            ordered_ids.append(summary.config_id)
    for summary in sorted(summaries, key=_summary_sort_key):
        if summary.config_id not in ordered_ids:
            ordered_ids.append(summary.config_id)
    by_id = {candidate.config_id: candidate for candidate in candidates}
    return [by_id[config_id] for config_id in ordered_ids[:limit] if config_id in by_id]


def _selection_roles(summaries: Sequence[OptimizerSummary]) -> dict[str, str]:
    latest = _latest_validation_summaries(summaries)
    if not latest:
        return {}
    safe = min(
        latest,
        key=lambda summary: (
            summary.false_fix_rate,
            -summary.fix_precision,
            -summary.correct_fix_rate,
            _optional_inf(summary.p95_position_error_m),
        ),
    )
    acceptable = [
        summary
        for summary in latest
        if summary.fix_precision >= 0.95 and summary.false_fix_rate <= safe.false_fix_rate + 1e-12
    ]
    if not acceptable:
        acceptable = latest
    fast = min(
        acceptable,
        key=lambda summary: (
            _optional_inf(summary.p95_tracking_time_ms),
            summary.false_fix_rate,
            -summary.correct_fix_rate,
        ),
    )
    accurate = min(
        latest,
        key=lambda summary: (
            _optional_inf(summary.p95_position_error_m),
            summary.false_fix_rate,
            -summary.correct_fix_rate,
        ),
    )
    balanced = max(
        latest,
        key=lambda summary: (
            summary.score,
            -summary.false_fix_rate,
            summary.correct_fix_rate,
            summary.fix_precision,
        ),
    )
    speed_free = min(
        latest,
        key=lambda summary: (
            _optional_inf(summary.speed_mae_m_s),
            summary.false_fix_rate,
            -summary.correct_fix_rate,
        ),
    )
    return {
        "safe": safe.config_id,
        "fast_acceptable": fast.config_id,
        "accurate": accurate.config_id,
        "balanced": balanced.config_id,
        "speed_free": speed_free.config_id,
    }


def _mark_summaries(
    summaries: Sequence[OptimizerSummary],
    pareto_ids: set[str],
    selected: dict[str, str],
) -> list[OptimizerSummary]:
    roles_by_id: dict[str, list[str]] = {}
    for role, config_id in selected.items():
        roles_by_id.setdefault(config_id, []).append(role)
    return [
        replace(
            summary,
            pareto_frontier=summary.config_id in pareto_ids,
            selection_role=",".join(sorted(roles_by_id.get(summary.config_id, []))),
        )
        for summary in summaries
    ]


def _pareto_frontier(
    summaries: Sequence[OptimizerSummary],
) -> list[OptimizerSummary]:
    latest = _latest_validation_summaries(summaries)
    frontier = []
    for candidate in latest:
        dominated = False
        for challenger in latest:
            if challenger.config_id == candidate.config_id:
                continue
            if _dominates(challenger, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=_summary_sort_key)


def _dominates(a: OptimizerSummary, b: OptimizerSummary) -> bool:
    a_values = (
        a.false_fix_rate,
        _optional_inf(a.p95_position_error_m),
        _optional_inf(a.p95_tracking_time_ms),
    )
    b_values = (
        b.false_fix_rate,
        _optional_inf(b.p95_position_error_m),
        _optional_inf(b.p95_tracking_time_ms),
    )
    return all(x <= y for x, y in zip(a_values, b_values, strict=True)) and any(
        x < y for x, y in zip(a_values, b_values, strict=True)
    )


def _latest_validation_summaries(
    summaries: Sequence[OptimizerSummary],
) -> list[OptimizerSummary]:
    best_by_id: dict[str, OptimizerSummary] = {}
    stage_order = {"coarse_sweep": 0, "refined_validation": 1, "unseen_final_test": 2}
    for summary in summaries:
        previous = best_by_id.get(summary.config_id)
        if previous is None or stage_order.get(summary.stage, -1) >= stage_order.get(previous.stage, -1):
            best_by_id[summary.config_id] = summary
    return list(best_by_id.values())


def _summary_by_id(
    summaries: Sequence[OptimizerSummary],
    config_id: str | None,
) -> OptimizerSummary | None:
    if config_id is None:
        return None
    matches = [summary for summary in summaries if summary.config_id == config_id]
    if not matches:
        return None
    return sorted(matches, key=lambda summary: summary.stage)[-1]


def _summary_sort_key(summary: OptimizerSummary) -> tuple[float, float, float, float, float, float, float]:
    return _summary_rank_tuple_values(
        summary.false_fix_rate,
        summary.correct_fix_rate,
        summary.fix_precision,
        summary.p95_position_error_m,
        summary.p95_tracking_time_ms,
        summary.initial_global_fix_time_ms,
        summary.speed_mae_m_s,
    )


def _summary_rank_tuple_values(
    false_rate: float,
    correct_rate: float,
    precision: float,
    p95_error: float | None,
    p95_tracking: float | None,
    initial_fix_runtime: float | None,
    speed_mae: float | None,
) -> tuple[float, float, float, float, float, float, float]:
    return (
        false_rate,
        -correct_rate,
        -precision,
        _optional_inf(p95_error),
        _optional_inf(p95_tracking),
        _optional_inf(initial_fix_runtime),
        _optional_inf(speed_mae),
    )


def _combined_score(
    *,
    correct_rate: float,
    false_rate: float,
    precision: float,
    ambiguous_rate: float,
    p95_error_m: float | None,
    p95_tracking_ms: float | None,
    speed_mae_m_s: float | None,
) -> float:
    p95_penalty = min(1.0, _optional_inf(p95_error_m) / DEFAULT_WRONG_FIX_THRESHOLD_M)
    runtime_penalty = min(1.0, _optional_inf(p95_tracking_ms) / 1000.0)
    speed_penalty = min(1.0, _optional_inf(speed_mae_m_s) / 5.0)
    return float(
        correct_rate
        + 0.20 * precision
        - 2.0 * false_rate
        - 0.25 * ambiguous_rate
        - 0.35 * p95_penalty
        - 0.10 * runtime_penalty
        - 0.10 * speed_penalty
    )


def _analysis_rows(
    summaries: Sequence[OptimizerSummary],
    fields: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[OptimizerSummary]] = {}
    for summary in _latest_validation_summaries(summaries):
        key = tuple(getattr(summary, field) for field in fields)
        groups.setdefault(key, []).append(summary)
    rows = []
    for key, group in groups.items():
        row = {field: value for field, value in zip(fields, key, strict=True)}
        row.update(_summary_group_metrics(group))
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["mean_score"]), float(row["mean_false_fix_rate"])))
    return rows


def _heading_analysis_rows(details: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in details:
        groups.setdefault(str(row.get("heading_scenario")), []).append(row)
    rows = []
    for heading, group in groups.items():
        total = len(group)
        rows.append(
            {
                "heading_scenario": heading,
                "updates": total,
                "correct_fix_rate": _rate(sum(1 for row in group if row.get("correct_fix")), total),
                "false_fix_rate": _rate(sum(1 for row in group if row.get("wrong_fix")), total),
                "fix_precision": _rate(
                    sum(1 for row in group if row.get("correct_fix")),
                    sum(1 for row in group if row.get("fix_accepted")),
                ),
                "speed_mae_m_s": _mean(row.get("speed_error_m_s") for row in group if row.get("fix_accepted")),
                "mean_runtime_ms": _mean(row.get("total_localization_ms") for row in group),
            }
        )
    return rows


def _runtime_breakdown_rows(summaries: Sequence[OptimizerSummary]) -> list[dict[str, object]]:
    rows = []
    for summary in _latest_validation_summaries(summaries):
        rows.append(
            {
                "config_id": summary.config_id,
                "coarse_speed_search_ms": summary.coarse_speed_search_ms,
                "medium_speed_search_ms": summary.medium_speed_search_ms,
                "fine_speed_search_ms": summary.fine_speed_search_ms,
                "dem_profile_matching_ms": summary.dem_profile_matching_ms,
                "quality_gate_ms": summary.quality_gate_ms,
                "position_ambiguity_ms": summary.position_ambiguity_ms,
                "speed_ambiguity_ms": summary.speed_ambiguity_ms,
                "mean_runtime_ms": summary.mean_runtime_ms,
                "p95_runtime_ms": summary.p95_runtime_ms,
            }
        )
    return rows


def _summary_group_metrics(group: Sequence[OptimizerSummary]) -> dict[str, object]:
    return {
        "count": len(group),
        "mean_score": _mean(summary.score for summary in group),
        "mean_correct_fix_rate": _mean(summary.correct_fix_rate for summary in group),
        "mean_false_fix_rate": _mean(summary.false_fix_rate for summary in group),
        "mean_fix_precision": _mean(summary.fix_precision for summary in group),
        "mean_p95_position_error_m": _mean(summary.p95_position_error_m for summary in group),
        "mean_p95_tracking_time_ms": _mean(summary.p95_tracking_time_ms for summary in group),
        "mean_speed_mae_m_s": _mean(summary.speed_mae_m_s for summary in group),
    }


def _default_vs_optimized_rows(
    default: OptimizerSummary,
    optimized: OptimizerSummary,
) -> list[dict[str, object]]:
    default_runtime = _optional_inf(default.mean_tracking_time_ms)
    optimized_runtime = _optional_inf(optimized.mean_tracking_time_ms)
    speedup = default_runtime / optimized_runtime if optimized_runtime > 0.0 else None
    return [
        {"metric": "DEFAULT UNKNOWN-SPEED Correct FIX", "value": default.correct_fix_rate},
        {"metric": "DEFAULT UNKNOWN-SPEED False FIX", "value": default.false_fix_rate},
        {"metric": "DEFAULT UNKNOWN-SPEED FIX Precision", "value": default.fix_precision},
        {"metric": "DEFAULT UNKNOWN-SPEED P95 Position Error", "value": default.p95_position_error_m},
        {"metric": "DEFAULT UNKNOWN-SPEED Speed MAE", "value": default.speed_mae_m_s},
        {"metric": "DEFAULT UNKNOWN-SPEED Initial Global Time", "value": default.initial_global_fix_time_ms},
        {"metric": "DEFAULT UNKNOWN-SPEED Mean Tracking Time", "value": default.mean_tracking_time_ms},
        {"metric": "DEFAULT UNKNOWN-SPEED P95 Tracking Time", "value": default.p95_tracking_time_ms},
        {"metric": "OPTIMIZED UNKNOWN-SPEED Correct FIX", "value": optimized.correct_fix_rate},
        {"metric": "OPTIMIZED UNKNOWN-SPEED False FIX", "value": optimized.false_fix_rate},
        {"metric": "OPTIMIZED UNKNOWN-SPEED FIX Precision", "value": optimized.fix_precision},
        {"metric": "OPTIMIZED UNKNOWN-SPEED P95 Position Error", "value": optimized.p95_position_error_m},
        {"metric": "OPTIMIZED UNKNOWN-SPEED Speed MAE", "value": optimized.speed_mae_m_s},
        {"metric": "OPTIMIZED UNKNOWN-SPEED Initial Global Time", "value": optimized.initial_global_fix_time_ms},
        {"metric": "OPTIMIZED UNKNOWN-SPEED Mean Tracking Time", "value": optimized.mean_tracking_time_ms},
        {"metric": "OPTIMIZED UNKNOWN-SPEED P95 Tracking Time", "value": optimized.p95_tracking_time_ms},
        {"metric": "CHANGE Speedup", "value": speedup},
        {"metric": "CHANGE Correct FIX", "value": optimized.correct_fix_rate - default.correct_fix_rate},
        {"metric": "CHANGE False FIX", "value": optimized.false_fix_rate - default.false_fix_rate},
        {"metric": "CHANGE Precision", "value": optimized.fix_precision - default.fix_precision},
        {
            "metric": "CHANGE Speed Error",
            "value": (
                optimized.speed_mae_m_s - default.speed_mae_m_s
                if optimized.speed_mae_m_s is not None and default.speed_mae_m_s is not None
                else None
            ),
        },
    ]


def _eliminated_rows(
    before: Sequence[OptimizerCandidate],
    after: Sequence[OptimizerCandidate],
    reason: str,
) -> list[dict[str, object]]:
    kept = {candidate.config_id for candidate in after}
    return [
        {**asdict(candidate), "eliminated_reason": reason}
        for candidate in before
        if candidate.config_id not in kept
    ]


def _numeric_values(values: Iterable[object]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            result.append(numeric)
    return result


def _rate(part: int, total: int) -> float:
    return float(part) / float(total) if total > 0 else 0.0


def _mean(values: Iterable[object]) -> float | None:
    clean = _numeric_values(values)
    return float(np.mean(clean)) if clean else None


def _median(values: Iterable[object]) -> float | None:
    clean = _numeric_values(values)
    return float(np.median(clean)) if clean else None


def _percentile(values: Iterable[object], percentile: float) -> float | None:
    clean = _numeric_values(values)
    return float(np.percentile(clean, percentile)) if clean else None


def _min_value(values: Iterable[object]) -> float | None:
    clean = _numeric_values(values)
    return min(clean) if clean else None


def _optional_inf(value: float | None) -> float:
    if value is None or not np.isfinite(value):
        return float("inf")
    return float(value)


def _format_optional(value: object) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "-"
    return f"{numeric:.1f} m"


def _format_optional_ms(value: object) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "-"
    return f"{numeric:.0f} ms"


def _should_stop(stop_requested: StopCallback | None) -> bool:
    return bool(stop_requested is not None and stop_requested())
