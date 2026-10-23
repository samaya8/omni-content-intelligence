"""Unit tests for the automated benchmark evaluation runner.

Tests coverage:
- Summarization benchmark execution and aggregate metrics
- Captioning benchmark execution and consensus metrics
- Full report generation to file and stdout
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from evals.runner import BenchmarkRunner


def test_summarization_benchmark():
    runner = BenchmarkRunner()
    results = runner.run_summarization_benchmark()

    assert "individual_runs" in results
    assert "aggregates" in results

    aggs = results["aggregates"]
    assert "extractive" in aggs
    assert "abstractive" in aggs

    ext = aggs["extractive"]
    assert ext["avg_rouge_1_f1"] > 0.0
    assert ext["avg_rouge_l_f1"] > 0.0
    assert ext["avg_compression_savings_pct"] > 0.0


def test_caption_benchmark():
    runner = BenchmarkRunner()
    results = runner.run_caption_benchmark()

    assert "individual_runs" in results
    assert "aggregates" in results

    aggs = results["aggregates"]
    assert "beam" in aggs
    assert "greedy" in aggs

    beam = aggs["beam"]
    assert beam["avg_bleu_1"] > 0.0
    assert beam["avg_confidence"] > 0.0
    assert beam["avg_cider"] >= 0.0


def test_report_generation(tmp_path: Path):
    runner = BenchmarkRunner()
    out_file = tmp_path / "TEST_BENCHMARK_RESULTS.md"
    report = runner.generate_report(output_markdown_path=str(out_file))

    assert "Enterprise Benchmark Report" in report
    assert "ROUGE" in report
    assert "BLEU" in report
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "Extractive (TextRank + MMR)" in content
    assert "Beam Search (K=3)" in content
