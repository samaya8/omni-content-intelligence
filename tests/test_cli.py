"""Unit tests for the Multimodal CLI suite.

Tests coverage:
- summarize subcommand (extractive, abstractive, bullet points, file inputs, JSON flags)
- caption subcommand (beam search, greedy, concept tags, JSON flags)
- evaluate subcommand (single/multi-reference, file inputs, metric filters, novelty metrics)
- batch processing (JSON manifest execution)
- Error handling and CLI exit codes
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
from PIL import Image

import cli


@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Fixture providing temporary files for CLI input/output tests."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "Transformers process sequential data via self-attention. "
        "They have scaled natural language processing to immense heights. "
        "Enterprise deployments require efficient evaluation and monitoring.",
        encoding="utf-8",
    )

    img_file = tmp_path / "sample.png"
    img = Image.new("RGB", (320, 240), color="blue")
    img.save(img_file)

    manifest_file = tmp_path / "manifest.json"
    manifest_data = [
        {
            "id": "task-1",
            "type": "summarize",
            "text": "Machine learning architectures are advancing rapidly. Automated summarization simplifies document review.",
            "strategy": "extractive",
            "style": "bullet_points",
        },
        {
            "id": "task-2",
            "type": "caption",
            "image": str(img_file),
            "beam_size": 2,
        },
    ]
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    return {
        "text_file": text_file,
        "img_file": img_file,
        "manifest_file": manifest_file,
        "tmp_path": tmp_path,
    }


def test_cli_help():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0


def test_cli_no_subcommand():
    code = cli.main([])
    assert code == 0


def test_cli_summarize_text_stdout(capsys):
    raw_text = (
        "Artificial intelligence is rapidly transforming global industry. "
        "Large language models automate repetitive data analysis workflows. "
        "Enterprises adopt these systems to increase overall productivity."
    )
    code = cli.main(["summarize", "--text", raw_text, "--strategy", "extractive", "--style", "bullet_points"])
    assert code == 0
    captured = capsys.readouterr()
    assert "SUMMARY" in captured.out
    assert "•" in captured.out


def test_cli_summarize_file_json(temp_workspace, capsys):
    fpath = str(temp_workspace["text_file"])
    code = cli.main(["summarize", "--file", fpath, "--strategy", "abstractive", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "summary" in payload
    assert "compression_ratio" in payload
    assert payload["strategy"].startswith("abstractive")


def test_cli_summarize_missing_file(capsys):
    code = cli.main(["summarize", "--file", "non_existent_file_12345.txt"])
    assert code == 1


def test_cli_caption_beam(temp_workspace, capsys):
    img_path = str(temp_workspace["img_file"])
    code = cli.main(["caption", "--image", img_path, "--method", "beam", "--beam-size", "3", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "caption" in payload
    assert payload["confidence"] > 0.0
    assert len(payload["candidate_beams"]) > 0


def test_cli_caption_greedy_stdout(temp_workspace, capsys):
    img_path = str(temp_workspace["img_file"])
    code = cli.main(["caption", "--image", img_path, "--method", "greedy", "--concepts"])
    assert code == 0
    captured = capsys.readouterr()
    assert "IMAGE CAPTION" in captured.out
    assert "Caption:" in captured.out


def test_cli_caption_missing_image():
    code = cli.main(["caption", "--image", "non_existent_image_9999.png"])
    assert code == 1


def test_cli_evaluate_inline(capsys):
    cand = "A cute puppy runs across the lawn."
    ref = "A small puppy is running on the green lawn."
    code = cli.main(["evaluate", "--candidate", cand, "--reference", ref, "--metric", "all", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "rouge" in payload
    assert "bleu" in payload
    assert "cider" in payload
    assert payload["rouge"]["rouge-1"]["f1"] > 0.0


def test_cli_evaluate_stdout(capsys):
    cand = "A cute puppy runs across the lawn."
    ref = "A small puppy is running on the green lawn."
    code = cli.main(["evaluate", "--candidate", cand, "--reference", ref])
    assert code == 0
    captured = capsys.readouterr()
    assert "NLG EVALUATION REPORT" in captured.out
    assert "ROUGE" in captured.out
    assert "BLEU" in captured.out


def test_cli_evaluate_files(temp_workspace, capsys):
    c_file = temp_workspace["tmp_path"] / "candidate.txt"
    r_file = temp_workspace["tmp_path"] / "reference.txt"
    c_file.write_text("Automated text summarization aids knowledge retrieval.", encoding="utf-8")
    r_file.write_text("Text summarization assists automated knowledge extraction.", encoding="utf-8")

    code = cli.main([
        "evaluate",
        "--candidate-file", str(c_file),
        "--reference-file", str(r_file),
        "--source", "Original raw source text for compression checks",
        "--json",
    ])
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "compression_and_novelty" in payload


def test_cli_batch_manifest(temp_workspace, capsys):
    manifest_path = str(temp_workspace["manifest_file"])
    out_file = str(temp_workspace["tmp_path"] / "results.json")

    code = cli.main(["batch", "--input", manifest_path, "--output", out_file, "--json"])
    assert code == 0
    assert os.path.exists(out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert "summary" in data[0]
    assert "caption" in data[1]
