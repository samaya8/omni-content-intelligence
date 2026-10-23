"""Enterprise Command-Line Interface (CLI) for Multimodal Content Intelligence.

Supports:
- summarize: Extractive, abstractive, or hybrid text summarization.
- caption: Visual token captioning with beam search or greedy decoding.
- evaluate: ROUGE, BLEU, CIDEr, and abstractiveness scoring against references.
- batch: Bulk processing of summarization and captioning tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from src.attention import MultimodalAttentionVisualizer
from src.captioner import ImageCaptioner
from src.eval_metrics import NLGEvalMetrics
from src.summarizer import TextSummarizer


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="multimodal-cli",
        description="Enterprise Multimodal Content Intelligence CLI (Summarization, Captioning, NLG Evals)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. Summarize
    p_sum = subparsers.add_parser("summarize", help="Summarize text input or file")
    p_sum.add_argument("--text", type=str, help="Raw text string to summarize")
    p_sum.add_argument("--file", type=str, help="Path to text file to summarize")
    p_sum.add_argument(
        "--strategy",
        "--method",
        dest="strategy",
        type=str,
        choices=["extractive", "abstractive", "hybrid"],
        default="extractive",
        help="Summarization strategy (extractive, abstractive, or hybrid)",
    )
    p_sum.add_argument(
        "--length",
        type=str,
        choices=["short", "medium", "long"],
        default="medium",
        help="Output length preset",
    )
    p_sum.add_argument(
        "--style",
        type=str,
        choices=["executive", "bullet_points", "dense"],
        default="executive",
        help="Formatting style of output",
    )
    p_sum.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 2. Caption
    p_cap = subparsers.add_parser("caption", help="Generate description for an image")
    p_cap.add_argument("--image", type=str, required=True, help="Path to image file")
    p_cap.add_argument(
        "--method",
        type=str,
        choices=["greedy", "beam"],
        default="beam",
        help="Decoding search strategy",
    )
    p_cap.add_argument("--beam-size", type=int, default=3, help="Beam width for beam search")
    p_cap.add_argument("--concepts", action="store_true", help="Extract visual concept tags")
    p_cap.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 3. Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute NLG evaluation metrics")
    p_eval.add_argument("--candidate", type=str, help="Candidate/generated text string")
    p_eval.add_argument("--candidate-file", type=str, help="Path to file containing candidate text")
    p_eval.add_argument(
        "--reference",
        type=str,
        action="append",
        dest="references",
        help="Reference ground-truth text (can be repeated)",
    )
    p_eval.add_argument("--reference-file", type=str, help="Path to file with reference text")
    p_eval.add_argument(
        "--metric",
        type=str,
        choices=["all", "rouge", "bleu", "cider"],
        default="all",
        help="Which metric family to calculate",
    )
    p_eval.add_argument("--source", type=str, help="Original source text for compression metrics")
    p_eval.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 4. Batch
    p_batch = subparsers.add_parser("batch", help="Batch process JSON manifest of tasks")
    p_batch.add_argument("--input", type=str, required=True, help="Input JSON manifest file path")
    p_batch.add_argument("--output", type=str, help="Output JSON results file path")
    p_batch.add_argument("--json", action="store_true", help="Print summary JSON to stdout")

    return parser


def handle_summarize(args: argparse.Namespace) -> int:
    """Execute summarize subcommand."""
    text = ""
    if args.text:
        text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Error: Either --text or --file must be specified for summarization.", file=sys.stderr)
        return 1

    summarizer = TextSummarizer()
    t0 = time.time()
    strategy = "extractive" if args.strategy == "hybrid" else args.strategy
    result = summarizer.summarize(
        text=text,
        strategy=strategy,
        length=args.length,
        format_style=args.style,
    )
    elapsed_ms = round((time.time() - t0) * 1000, 2)

    summary_text = result.get("summary", "")
    key_points = result.get("sentences", [])
    compression_ratio = result.get("compression_ratio", 1.0)
    savings_pct = round((1.0 - compression_ratio) * 100, 1)

    payload = {
        "strategy": result.get("strategy", args.strategy),
        "style": args.style,
        "summary": summary_text,
        "key_points": key_points,
        "compression_ratio": compression_ratio,
        "savings_pct": savings_pct,
        "latency_ms": elapsed_ms,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"SUMMARY ({args.strategy.upper()} - {args.style.upper()})")
        print("=" * 60)
        print(summary_text)
        if key_points:
            print("\nKey Extracted Sentences:")
            for idx, pt in enumerate(key_points, 1):
                print(f"  {idx}. {pt}")
        print(f"\nCompression: {compression_ratio} ({savings_pct}% reduction) in {elapsed_ms}ms")
        print("=" * 60)

    return 0


def handle_caption(args: argparse.Namespace) -> int:
    """Execute caption subcommand."""
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 1

    captioner = ImageCaptioner()
    t0 = time.time()
    beam_size = 1 if args.method == "greedy" else max(1, args.beam_size)
    result = captioner.generate_caption(
        image_input=args.image,
        beam_size=beam_size,
    )
    elapsed_ms = round((time.time() - t0) * 1000, 2)

    payload = {
        "image": args.image,
        "caption": result["caption"],
        "confidence": result["confidence"],
        "decoding_method": result["decoding_method"],
        "detected_entities": result.get("detected_entities", {}),
        "candidate_beams": result.get("candidate_beams", []),
        "latency_ms": elapsed_ms,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"IMAGE CAPTION ({result['decoding_method'].upper()})")
        print("=" * 60)
        print(f"Caption:    {result['caption']}")
        print(f"Confidence: {round(result['confidence'] * 100, 1)}%")
        entities = result.get("detected_entities", {})
        if entities:
            print("Visual Concepts:")
            for category, item in entities.items():
                if item:
                    val = ", ".join(item) if isinstance(item, list) else str(item)
                    print(f"  - {category.replace('_', ' ').capitalize()}: {val}")
        print(f"Latency:    {elapsed_ms}ms")
        print("=" * 60)

    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """Execute evaluate subcommand."""
    candidate = ""
    if args.candidate:
        candidate = args.candidate
    elif args.candidate_file:
        if not os.path.exists(args.candidate_file):
            print(f"Error: Candidate file not found: {args.candidate_file}", file=sys.stderr)
            return 1
        with open(args.candidate_file, "r", encoding="utf-8") as f:
            candidate = f.read().strip()
    else:
        print("Error: Either --candidate or --candidate-file is required.", file=sys.stderr)
        return 1

    references: List[str] = args.references or []
    if args.reference_file:
        if not os.path.exists(args.reference_file):
            print(f"Error: Reference file not found: {args.reference_file}", file=sys.stderr)
            return 1
        with open(args.reference_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            references.extend(lines)

    if not references:
        print("Error: At least one --reference or --reference-file must be provided.", file=sys.stderr)
        return 1

    eval_results: Dict[str, Any] = {"candidate": candidate, "references": references}

    primary_ref = references[0]

    if args.metric in ("all", "rouge"):
        eval_results["rouge"] = NLGEvalMetrics.compute_rouge(candidate, primary_ref)

    if args.metric in ("all", "bleu"):
        eval_results["bleu"] = NLGEvalMetrics.compute_bleu(candidate, references)

    if args.metric in ("all", "cider"):
        eval_results["cider"] = NLGEvalMetrics.compute_cider(candidate, references)

    if args.source:
        eval_results["compression_and_novelty"] = NLGEvalMetrics.compute_abstractiveness_and_compression(
            candidate, args.source
        )

    if args.json:
        print(json.dumps(eval_results, indent=2))
    else:
        print("\n" + "=" * 60)
        print("NLG EVALUATION REPORT")
        print("=" * 60)
        print(f"Candidate:  {candidate}")
        print(f"References: {len(references)} reference(s)")

        if "rouge" in eval_results:
            print("\n[ROUGE SCORES]")
            for k, scores in eval_results["rouge"].items():
                print(f"  {k.upper():<8} P: {scores['precision']:>5.1f}% | R: {scores['recall']:>5.1f}% | F1: {scores['f1']:>5.1f}%")

        if "bleu" in eval_results:
            print("\n[BLEU SCORES]")
            b = eval_results["bleu"]
            print(f"  BP: {b['brevity_penalty']} | BLEU-1: {b['bleu-1']}% | BLEU-2: {b['bleu-2']}% | BLEU-3: {b['bleu-3']}% | BLEU-4: {b['bleu-4']}%")

        if "cider" in eval_results:
            print(f"\n[CIDEr CONSENSUS]: {eval_results['cider']}")

        if "compression_and_novelty" in eval_results:
            cn = eval_results["compression_and_novelty"]
            print(f"\n[COMPRESSION & NOVELTY]")
            print(f"  Compression Ratio: {cn['compression_ratio']} ({cn['compression_savings_pct']}% savings)")
            print(f"  Novel Unigram:     {cn['novel_unigram_ratio']}% | Novel Bigram: {cn['novel_bigram_ratio']}%")

        print("=" * 60)

    return 0


def handle_batch(args: argparse.Namespace) -> int:
    """Execute batch manifest processing."""
    if not os.path.exists(args.input):
        print(f"Error: Manifest file not found: {args.input}", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list):
        print("Error: Input manifest must be a JSON list of tasks.", file=sys.stderr)
        return 1

    summarizer = TextSummarizer()
    captioner = ImageCaptioner()

    results: List[Dict[str, Any]] = []

    for idx, task in enumerate(manifest):
        task_type = task.get("type", "summarize")
        item_res: Dict[str, Any] = {"id": task.get("id", idx), "type": task_type}

        if task_type == "summarize":
            text = task.get("text", "")
            strategy = task.get("strategy", task.get("method", "extractive"))
            style = task.get("style", "executive")
            length = task.get("length", "medium")
            res = summarizer.summarize(text=text, strategy=strategy, length=length, format_style=style)
            item_res["summary"] = res.get("summary", "")
            item_res["sentences"] = res.get("sentences", [])
            item_res["compression_ratio"] = res.get("compression_ratio", 1.0)

        elif task_type == "caption":
            image_path = task.get("image", "")
            beam_size = task.get("beam_size", 3)
            if os.path.exists(image_path):
                cap_res = captioner.generate_caption(image_path, beam_size=beam_size)
                item_res["caption"] = cap_res["caption"]
                item_res["confidence"] = cap_res["confidence"]
                item_res["detected_entities"] = cap_res.get("detected_entities", {})
            else:
                item_res["error"] = f"Image path not found: {image_path}"
        else:
            item_res["error"] = f"Unknown task type: {task_type}"

        results.append(item_res)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Processed {len(results)} tasks. Output saved to {args.output}")

    if args.json or not args.output:
        print(json.dumps(results, indent=2))

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "summarize":
        return handle_summarize(args)
    elif args.subcommand == "caption":
        return handle_caption(args)
    elif args.subcommand == "evaluate":
        return handle_evaluate(args)
    elif args.subcommand == "batch":
        return handle_batch(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
