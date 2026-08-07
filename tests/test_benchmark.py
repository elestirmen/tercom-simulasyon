"""Tests for the offline profile-vector benchmark mode."""

import zipfile
from dataclasses import replace

from terrain_nav.benchmark import (
    benchmark_overview_rows,
    benchmark_variant_summary_rows,
    build_benchmark_routes,
    default_benchmark_variants,
    format_benchmark_summary,
    run_benchmark_suite,
)
from terrain_nav.config import LocalizationConfig
from terrain_nav.terrain import TerrainManager


def _benchmark_config() -> LocalizationConfig:
    config = LocalizationConfig()
    terrain = replace(
        config.terrain,
        rows=80,
        cols=80,
        dem_noise_std_m=0.0,
        dem_bias_m=0.0,
    )
    sensor = replace(
        config.sensor,
        laser_noise_std_m=0.0,
        laser_bias_m=0.0,
        laser_outlier_prob=0.0,
        laser_drop_prob=0.0,
    )
    algorithm = replace(
        config.algorithm,
        min_profile_length=3,
        coarse_stride=4,
        medium_stride=2,
        top_k=3,
        loss_method="rmse",
    )
    return replace(config, terrain=terrain, sensor=sensor, algorithm=algorithm)


def test_benchmark_suite_compares_raw_and_interpolated_vectors(tmp_path):
    result = run_benchmark_suite(
        _benchmark_config(),
        point_counts=(6, 10),
        checkpoint_fractions=(0.5, 1.0),
        max_routes=2,
        dense_sample_count=10,
        output_dir=tmp_path,
    )

    assert len(result.route_paths) == 2
    assert len(result.summaries) == 16
    assert {summary.variant_name for summary in result.summaries} == {
        "raw_6",
        "interp_6",
        "raw_10",
        "interp_10",
    }
    assert {summary.checkpoint_fraction for summary in result.summaries} == {0.5, 1.0}
    assert all(2 <= summary.used_points <= summary.requested_points for summary in result.summaries)
    assert all(summary.elapsed_ms >= 0.0 for summary in result.summaries)
    assert result.summary_csv_path is not None
    assert result.details_jsonl_path is not None
    assert result.excel_path is not None
    with zipfile.ZipFile(result.excel_path) as workbook:
        names = set(workbook.namelist())
    assert "xl/workbook.xml" in names
    assert "xl/worksheets/sheet1.xml" in names
    assert "xl/worksheets/sheet5.xml" in names
    assert any(row["metric"] == "Excel dosyasi" for row in benchmark_overview_rows(result))
    assert len(benchmark_variant_summary_rows(result)) == 4
    assert "P95" in format_benchmark_summary(result)


def test_default_benchmark_plan_is_comprehensive():
    terrain = TerrainManager(_benchmark_config().terrain)
    try:
        routes = build_benchmark_routes(terrain, max_routes=32)
    finally:
        terrain.close()

    assert len(routes) == 32
    assert len(default_benchmark_variants()) == 14
