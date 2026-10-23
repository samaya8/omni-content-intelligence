"""Multimodal Intelligence Lab & Evaluation Arena (Streamlit Advanced Workbench).

Features:
- Tab 1: Summarization Arena (Extractive vs. Abstractive Head-to-Head with ROUGE & Novelty metrics).
- Tab 2: Beam Search Visualizer & Attention Explorer (Token-by-token spatial cross-attention inspection).
- Tab 3: NLG Evaluation Calculator (Interactive ROUGE, BLEU-1..4, and CIDEr consensus scorer).
- Tab 4: Batch Processing Queue & Manifest Runner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

from src.attention import MultimodalAttentionVisualizer
from src.captioner import ImageCaptioner
from src.eval_metrics import NLGEvalMetrics
from src.summarizer import TextSummarizer


st.set_page_config(
    page_title="OmniContent AI | NLP Analytics Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize engines
if "summarizer" not in st.session_state:
    st.session_state.summarizer = TextSummarizer()
if "captioner" not in st.session_state:
    st.session_state.captioner = ImageCaptioner()

with st.sidebar:
    st.title("🔬 Lab Workbench")
    st.markdown("Advanced multimodal diagnostic, comparative, and evaluation instruments.")
    st.info("Zero-GPU edge execution enabled • Deterministic reproducible benchmarks")
    st.divider()
    st.markdown("### Quick Benchmark Presets")
    st.markdown("- **ROUGE-L**: Dynamic programming LCS")
    st.markdown("- **BLEU-1..4**: Modified n-gram with Brevity Penalty")
    st.markdown("- **CIDEr**: Consensus TF-IDF cosine similarity")
    st.divider()
    st.caption("Multimodal Content Intelligence • Evaluation Suite v2.0")

st.title("🔬 Multimodal Intelligence Lab & Evaluation Arena")
st.markdown(
    "Enterprise diagnostics workbench for multi-model ablation, visual cross-attention exploration, "
    "and pure-Python Natural Language Generation (NLG) benchmark calculation."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "⚔️ Summarization Arena",
    "🎯 Beam Visualizer & Attention",
    "📊 NLG Benchmark Calculator",
    "📦 Batch Processing Queue",
])


# =========================================================
# TAB 1: SUMMARIZATION ARENA (HEAD-TO-HEAD ABLATION)
# =========================================================
with tab1:
    st.markdown("### Extractive (TextRank) vs. Abstractive (Neural) Arena")
    st.caption("Run both models simultaneously on the same text corpus to compare compression, abstractiveness, and cross-similarity.")

    sample_doc = (
        "Large language models and vision-language architectures are transforming automated enterprise operations. "
        "Unlike previous recurrent neural networks, the transformer relies exclusively on self-attention mechanisms to "
        "draw global dependencies between inputs and outputs. This architecture enables significantly higher degree of parallelization "
        "during model training on modern high-performance clusters. Extractive summarization methods construct sentence graphs "
        "and select central nodes via PageRank and Maximal Marginal Relevance to eliminate redundant passages. "
        "In contrast, abstractive systems formulate paraphrased synthesized narratives that integrate multiple concepts. "
        "Empirical benchmarks demonstrate that combining extractive filtering with abstractive refinement yields superior factual accuracy."
    )

    arena_text = st.text_area("Corpus Text for Arena Evaluation:", value=sample_doc, height=160)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_arena = st.button("⚔️ Run Head-to-Head Comparison", type="primary")

    if run_arena and arena_text.strip():
        with st.spinner("Executing dual-pipeline inference and computing ablation metrics..."):
            # Extractive
            t0 = time.time()
            ext_res = st.session_state.summarizer.summarize(arena_text, strategy="extractive", length="medium")
            ext_latency = round((time.time() - t0) * 1000, 1)

            # Abstractive
            t1 = time.time()
            abs_res = st.session_state.summarizer.summarize(arena_text, strategy="abstractive", length="medium")
            abs_latency = round((time.time() - t1) * 1000, 1)

            # Comparative metrics
            ext_metrics = NLGEvalMetrics.compute_abstractiveness_and_compression(ext_res["summary"], arena_text)
            abs_metrics = NLGEvalMetrics.compute_abstractiveness_and_compression(abs_res["summary"], arena_text)

            # Cross-model agreement (ROUGE-L between summaries)
            cross_rouge = NLGEvalMetrics.compute_rouge(ext_res["summary"], abs_res["summary"])

            st.markdown("#### ⚖️ Side-by-Side Performance Matrix")
            col_ext, col_abs = st.columns(2)

            with col_ext:
                st.markdown("##### 📌 Extractive Strategy (TextRank + MMR)")
                st.info(ext_res["summary"])
                e1, e2, e3 = st.columns(3)
                e1.metric("Compression", f"{ext_metrics['compression_savings_pct']}% saved")
                e2.metric("Novel Unigrams", f"{ext_metrics['novel_unigram_ratio']}%")
                e3.metric("Latency", f"{ext_latency} ms")

            with col_abs:
                st.markdown("##### ✨ Abstractive Strategy (Neural Synthesizer)")
                st.info(abs_res["summary"])
                a1, a2, a3 = st.columns(3)
                a1.metric("Compression", f"{abs_metrics['compression_savings_pct']}% saved")
                a2.metric("Novel Unigrams", f"{abs_metrics['novel_unigram_ratio']}%")
                a3.metric("Latency", f"{abs_latency} ms")

            st.markdown("#### 🤝 Inter-Model Consensus & Agreement")
            st.write(
                f"**ROUGE-L F1 Agreement**: `{cross_rouge['rouge-l']['f1']}%` | "
                f"**ROUGE-1 F1 Agreement**: `{cross_rouge['rouge-1']['f1']}%` | "
                f"**ROUGE-2 F1 Agreement**: `{cross_rouge['rouge-2']['f1']}%`"
            )


# =========================================================
# TAB 2: BEAM SEARCH VISUALIZER & ATTENTION EXPLORER
# =========================================================
with tab2:
    st.markdown("### Beam Search Tree & Spatial Cross-Attention Explorer")
    st.caption("Inspect how the visual encoder projects spatial weights across tokens during sequential generation.")

    col_img_up, col_img_ctrl = st.columns([1, 1])
    with col_img_up:
        img_upload = st.file_uploader("Upload Image to Inspect Attention:", type=["png", "jpg", "jpeg", "webp"], key="attn_img")
        sample_img = Image.new("RGB", (360, 270), color=(60, 110, 180)) if img_upload is None else Image.open(img_upload)
        st.image(sample_img, caption="Analyzed Image", use_container_width=True)

    with col_img_ctrl:
        st.markdown("##### Decoding Parameters")
        k_beams = st.slider("Beam Search Width (K)", min_value=1, max_value=5, value=3, key="attn_beams")
        run_attn = st.button("🔍 Generate & Analyze Beams", type="primary")

    if run_attn or "last_cap_res" in st.session_state:
        if run_attn:
            st.session_state.last_cap_res = st.session_state.captioner.generate_caption(sample_img, beam_size=k_beams)

        res = st.session_state.last_cap_res
        st.markdown(f"**Top Caption**: *\"{res['caption']}\"* (Confidence: `{round(res['confidence']*100, 1)}%`)")

        # Beam hypotheses
        st.markdown("#### 🌲 Beam Search Hypothesis Tree")
        beams = res.get("candidate_beams", [])
        st.dataframe(beams, use_container_width=True)

        # Token-level spatial attention maps
        st.markdown("#### 🗺️ Token-by-Token Spatial Cross-Attention Maps")
        tokens = res["caption"].split()
        selected_token = st.selectbox("Select Generated Token for Attention Inspection:", tokens, index=min(1, len(tokens)-1))

        heatmap_matrix = MultimodalAttentionVisualizer.generate_spatial_attention_map(sample_img, token=selected_token, grid_size=8)

        col_heat, col_stats = st.columns([2, 1])
        with col_heat:
            st.write(f"**8x8 Spatial Cross-Attention Density for token: `{selected_token}`**")
            st.dataframe(heatmap_matrix, use_container_width=True)
        with col_stats:
            flat_vals = [val for row in heatmap_matrix for val in row]
            st.metric("Peak Attention", str(max(flat_vals)))
            st.metric("Mean Spatial Attention", str(round(sum(flat_vals)/len(flat_vals), 4)))
            st.metric("Attention Entropy", str(round(-sum(p * 0.01 for p in flat_vals[:10]), 3)))


# =========================================================
# TAB 3: NLG BENCHMARK CALCULATOR
# =========================================================
with tab3:
    st.markdown("### Pure-Python NLG Benchmark Calculator")
    st.caption("Calculate exact ROUGE-1/2/L, BLEU-1/2/3/4 with Brevity Penalty, and CIDEr consensus metrics.")

    col_cand, col_ref = st.columns(2)
    with col_cand:
        cand_input = st.text_area(
            "Candidate Generated Text:",
            value="A dark dog running through the grassy field in sunny weather.",
            height=120,
        )
    with col_ref:
        ref_input = st.text_area(
            "Reference Descriptions (one per line):",
            value=(
                "A black dog running through the green grass field.\n"
                "A happy dark canine plays on the lawn outside.\n"
                "A puppy sprints across the sunny grass park."
            ),
            height=120,
        )

    calc_btn = st.button("📊 Calculate NLG Metrics", type="primary")

    if calc_btn and cand_input.strip() and ref_input.strip():
        refs = [r.strip() for r in ref_input.splitlines() if r.strip()]

        rouge_scores = NLGEvalMetrics.compute_rouge(cand_input, refs[0])
        bleu_scores = NLGEvalMetrics.compute_bleu(cand_input, refs)
        cider_score = NLGEvalMetrics.compute_cider(cand_input, refs)

        st.markdown("#### 🎯 Benchmark Metric Dashboard")

        # ROUGE
        st.markdown("##### ROUGE Scores (vs. Primary Reference)")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("ROUGE-1 F1", f"{rouge_scores['rouge-1']['f1']}%", delta=f"P: {rouge_scores['rouge-1']['precision']}% | R: {rouge_scores['rouge-1']['recall']}%")
        rc2.metric("ROUGE-2 F1", f"{rouge_scores['rouge-2']['f1']}%", delta=f"P: {rouge_scores['rouge-2']['precision']}% | R: {rouge_scores['rouge-2']['recall']}%")
        rc3.metric("ROUGE-L F1", f"{rouge_scores['rouge-l']['f1']}%", delta=f"P: {rouge_scores['rouge-l']['precision']}% | R: {rouge_scores['rouge-l']['recall']}%")

        # BLEU & CIDEr
        st.markdown("##### BLEU & CIDEr Scores (Multi-Reference Consensus)")
        bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(6)
        bc1.metric("BLEU-1", f"{bleu_scores['bleu-1']}%")
        bc2.metric("BLEU-2", f"{bleu_scores['bleu-2']}%")
        bc3.metric("BLEU-3", f"{bleu_scores['bleu-3']}%")
        bc4.metric("BLEU-4", f"{bleu_scores['bleu-4']}%")
        bc5.metric("Brevity Penalty", str(bleu_scores["brevity_penalty"]))
        bc6.metric("CIDEr Score", str(cider_score))


# =========================================================
# TAB 4: BATCH PROCESSING QUEUE
# =========================================================
with tab4:
    st.markdown("### Bulk Task Execution Queue")
    st.caption("Upload JSON task manifests to process document collections and image datasets in high-throughput batch.")

    sample_manifest = [
        {"id": "doc-001", "type": "summarize", "text": "Deep learning models transform enterprise automation.", "strategy": "extractive", "length": "short"},
        {"id": "doc-002", "type": "summarize", "text": "Transformers process text sequences with multi-head attention mechanisms.", "strategy": "abstractive", "length": "short"},
    ]

    manifest_upload = st.file_uploader("Upload JSON Task Manifest:", type=["json"])
    manifest_data = None

    if manifest_upload is not None:
        try:
            manifest_data = json.load(manifest_upload)
            st.success(f"Loaded manifest with {len(manifest_data)} tasks.")
        except Exception as e:
            st.error(f"Error parsing manifest JSON: {e}")
    else:
        st.caption("No file uploaded. Click below to load sample 2-task manifest.")
        if st.button("Load Sample Manifest"):
            manifest_data = sample_manifest

    if manifest_data is not None:
        if st.button("🚀 Process Batch Queue", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            batch_results = []

            for idx, task in enumerate(manifest_data):
                status_text.text(f"Processing task {idx+1}/{len(manifest_data)}: {task.get('id', idx)}...")
                task_type = task.get("type", "summarize")
                res_item = {"id": task.get("id", idx), "type": task_type}

                if task_type == "summarize":
                    s_out = st.session_state.summarizer.summarize(
                        text=task.get("text", ""),
                        strategy=task.get("strategy", "extractive"),
                        length=task.get("length", "medium"),
                    )
                    res_item["summary"] = s_out["summary"]
                    res_item["compression_ratio"] = s_out["compression_ratio"]
                elif task_type == "caption":
                    c_out = st.session_state.captioner.generate_caption(
                        image_input=task.get("image", "sample.png"),
                    )
                    res_item["caption"] = c_out["caption"]
                    res_item["confidence"] = c_out["confidence"]

                batch_results.append(res_item)
                progress_bar.progress((idx + 1) / len(manifest_data))

            status_text.text("Batch execution complete!")
            st.dataframe(batch_results, use_container_width=True)

            out_json = json.dumps(batch_results, indent=2)
            st.download_button(
                "💾 Download Batch Results (.json)",
                data=out_json,
                file_name="batch_results.json",
                mime="application/json",
            )
