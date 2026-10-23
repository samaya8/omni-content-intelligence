"""Enterprise Multimodal Content Intelligence Studio (Streamlit Application).

Provides:
- Tab 1: Dual-Strategy Text Summarization with salience highlighting and KPI metrics.
- Tab 2: Vision-Language Image Captioning with concept entity tagging and attention overlay.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

from src.attention import MultimodalAttentionVisualizer
from src.captioner import ImageCaptioner
from src.eval_metrics import NLGEvalMetrics
from src.summarizer import TextSummarizer


st.set_page_config(
    page_title="OmniContent AI | Speech-to-Insight Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "summarizer" not in st.session_state:
    st.session_state.summarizer = TextSummarizer()
if "captioner" not in st.session_state:
    st.session_state.captioner = ImageCaptioner()

# Preloaded sample texts for instant demonstration
SAMPLE_TEXTS = {
    "AI Enterprise Whitepaper": (
        "Artificial intelligence and deep learning architectures have revolutionized enterprise information workflows. "
        "Transformer-based foundations leverage self-attention mechanisms to analyze textual and multimodal content "
        "at unprecedented scale. By eliminating recurrence, transformers enable distributed matrix computations across modern "
        "GPU clusters, dramatically decreasing training latency. Extractive summarization identifies central graph nodes within "
        "document sentence graphs using eigenvector centrality. Meanwhile, abstractive pipelines synthesize novel contextual paraphrases "
        "while enforcing semantic coherence. Organizations adopting dual-strategy document intelligence report 65% faster review "
        "cycles and enhanced knowledge discovery across technical repositories."
    ),
    "Quarterly Financial Briefing": (
        "Global Cloud Solutions announced record fourth-quarter fiscal results, driven by strong enterprise demand for generative AI. "
        "Total revenue grew 28% year-over-year to $4.8 billion, exceeding analyst projections across all operational units. "
        "The Infrastructure and Multimodal Services segment saw the fastest growth, expanding by 42% compared to the previous period. "
        "Operating margins improved to 34%, supported by strategic cost optimizations in data center energy management. "
        "Management raised full-year forward guidance, projecting continued expansion in enterprise automated indexing solutions."
    ),
    "Scientific Climate Report": (
        "Satellite observations and oceanographic sensor networks indicate accelerated melting in sub-polar ice sheets. "
        "Mean surface temperatures over high-latitude zones increased by 1.4 degrees Celsius over the preceding decade. "
        "Oceanic heat content reached unprecedented levels, disrupting marine biological ecosystems and altering coastal currents. "
        "Atmospheric monitoring reveals greenhouse gas concentrations surpassing 420 parts per million at baseline observatories. "
        "International collaborative climate models emphasize that coordinated emission reductions are critical to mitigating severe global impacts."
    ),
}

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Engine Controls")
    st.markdown("Configure multimodal inference parameters and model styles.")

    st.subheader("📝 Summarizer Options")
    sum_strategy = st.selectbox(
        "Summarization Strategy",
        ["extractive", "abstractive"],
        index=0,
        help="Extractive uses graph-based TextRank + MMR; Abstractive synthesizes fluent narrative digests.",
    )
    sum_length = st.select_slider(
        "Length Preset",
        options=["short", "medium", "long"],
        value="medium",
        help="Target length: short (2 sentences), medium (4 sentences), long (6 sentences).",
    )
    sum_style = st.selectbox(
        "Presentation Style",
        ["executive", "bullet_points", "dense"],
        index=0,
        help="Format structure for the generated digest.",
    )

    st.divider()

    st.subheader("🖼️ Captioner Options")
    cap_strategy = st.radio("Decoding Search", ["Beam Search", "Greedy Search"], index=0)
    cap_beam_size = st.slider("Beam Width (K)", min_value=1, max_value=5, value=3) if cap_strategy == "Beam Search" else 1
    show_attention_map = st.checkbox("Generate Spatial Attention Map", value=True)

    st.divider()
    st.caption("Multimodal Content Intelligence Platform v2.0 • Offline Reproducible • Zero GPU Required")


# --- Main Application Header ---
st.title("🧠 Multimodal Content Intelligence Studio")
st.markdown(
    "Enterprise-grade dual-modal generation suite combining **Graph & Neural Text Summarization** "
    "with **Vision-Language Scene Captioning**, explainable attention maps, and standard NLG benchmark metrics."
)

tab_sum, tab_cap = st.tabs(["📝 Text Summarization Studio", "🖼️ Vision-Language Captioning"])


# ==========================================
# TAB 1: TEXT SUMMARIZATION
# ==========================================
with tab_sum:
    st.markdown("### Document Ingestion & Dual-Strategy Summarizer")

    col_input_ctrl, col_sample = st.columns([2, 1])
    with col_sample:
        selected_sample = st.selectbox("Load Curated Enterprise Sample:", ["(Custom Input)"] + list(SAMPLE_TEXTS.keys()))

    text_default = SAMPLE_TEXTS.get(selected_sample, "") if selected_sample != "(Custom Input)" else ""

    text_input = st.text_area(
        "Input Document / Source Text:",
        value=text_default,
        height=180,
        placeholder="Paste document text, research abstract, or memo here...",
    )

    uploaded_file = st.file_uploader("Or upload text document (.txt):", type=["txt"], key="sum_upload")
    if uploaded_file is not None:
        try:
            text_input = uploaded_file.read().decode("utf-8")
            st.success(f"Uploaded `{uploaded_file.name}` ({len(text_input)} characters)")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_sum = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

    if run_sum and text_input.strip():
        with st.spinner("Analyzing document graph and generating summary..."):
            t0 = time.time()
            res = st.session_state.summarizer.summarize(
                text=text_input,
                strategy=sum_strategy,
                length=sum_length,
                format_style=sum_style,
            )
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            orig_words = len(text_input.split())
            sum_words = len(res["summary"].split())
            comp_ratio = res["compression_ratio"]
            savings_pct = round((1.0 - comp_ratio) * 100, 1)

            st.success(f"Summary generated in **{elapsed_ms} ms** using **{res.get('strategy', sum_strategy).upper()}**!")

            # Metric KPIs
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original Length", f"{orig_words} words")
            m2.metric("Summary Length", f"{sum_words} words")
            m3.metric("Reduction Ratio", f"{savings_pct}%", delta=f"{comp_ratio}x comp")
            m4.metric("Inference Latency", f"{elapsed_ms} ms")

            st.markdown("#### 📄 Executive Summary Output")
            st.info(res["summary"])

            # Highlighted Salience View for Extractive
            if "sentences" in res and res["sentences"]:
                st.markdown("#### 🔍 Explainable Text Salience Heatmap")
                st.caption("Original text with top-ranked salient sentences highlighted based on TextRank PageRank scores:")
                sentences = st.session_state.summarizer.split_sentences(text_input)
                scores = st.session_state.summarizer._compute_textrank_scores(sentences)
                highlight_html = MultimodalAttentionVisualizer.create_salience_html(sentences, scores)
                st.components.v1.html(highlight_html, height=180, scrolling=True)

            # Export actions
            exp_col1, exp_col2, _ = st.columns([1, 1, 2])
            with exp_col1:
                st.download_button(
                    "💾 Download Summary (.txt)",
                    data=res["summary"],
                    file_name="summary_output.txt",
                    mime="text/plain",
                )
            with exp_col2:
                json_data = json.dumps(res, indent=2)
                st.download_button(
                    "📦 Export Metadata (.json)",
                    data=json_data,
                    file_name="summary_metadata.json",
                    mime="application/json",
                )


# ==========================================
# TAB 2: IMAGE CAPTIONING
# ==========================================
with tab_cap:
    st.markdown("### Vision-Language Image Captioning & Spatial Attention")

    col_img_in, col_img_disp = st.columns([1, 1])

    with col_img_in:
        uploaded_img = st.file_uploader("Upload Image File:", type=["png", "jpg", "jpeg", "webp", "bmp"])
        use_sample = st.checkbox("Or use standard visual test pattern", value=uploaded_img is None)

        img_to_process = None
        if uploaded_img is not None:
            try:
                img_to_process = Image.open(uploaded_img)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
        elif use_sample:
            # Generate deterministic procedural color canvas
            img_to_process = Image.new("RGB", (400, 300), color=(45, 95, 160))

        run_caption = st.button("👁️ Generate Visual Caption", type="primary", use_container_width=True)

    with col_img_disp:
        if img_to_process is not None:
            st.image(img_to_process, caption="Active Image Input", use_container_width=True)

    if run_caption and img_to_process is not None:
        with st.spinner("Decoding visual tokens and executing beam search..."):
            t0 = time.time()
            cap_res = st.session_state.captioner.generate_caption(
                image_input=img_to_process,
                beam_size=cap_beam_size,
            )
            cap_latency_ms = round((time.time() - t0) * 1000, 1)

            st.success(f"Caption generated in **{cap_latency_ms} ms** via **{cap_res['decoding_method']}**!")

            c1, c2, c3 = st.columns(3)
            c1.metric("Top Caption Confidence", f"{round(cap_res['confidence'] * 100, 1)}%")
            c2.metric("Beam Width", str(cap_beam_size))
            c3.metric("Latency", f"{cap_latency_ms} ms")

            st.markdown("#### 💬 Generated Image Description")
            st.info(f"**\"{cap_res['caption']}\"**")

            # Visual Concepts Tag Cloud
            entities = cap_res.get("detected_entities", {})
            if entities:
                st.markdown("#### 🏷️ Detected Conceptual Attributes")
                cols = st.columns(len(entities))
                for idx, (cat, val) in enumerate(entities.items()):
                    with cols[idx]:
                        st.markdown(f"**{cat.replace('_', ' ').capitalize()}**")
                        display_val = ", ".join(val) if isinstance(val, list) else str(val)
                        st.code(display_val, language="markdown")

            # Candidate Beams
            beams = cap_res.get("candidate_beams", [])
            if len(beams) > 1:
                st.markdown("#### 🌲 Candidate Search Beams")
                beam_table = [
                    {"Rank": idx + 1, "Hypothesis Caption": b["caption"], "Score": b["score"], "Confidence": f"{round(b['confidence']*100, 1)}%"}
                    for idx, b in enumerate(beams)
                ]
                st.dataframe(beam_table, use_container_width=True)

            # Spatial Attention Overlay
            if show_attention_map:
                st.markdown("#### 🎯 Visual Attention Density Map (8x8 Spatial Grid)")
                st.caption("Normalized spatial cross-attention heatmap for the key generated visual token:")
                tokens = cap_res["caption"].split()
                key_token = tokens[1] if len(tokens) > 1 else tokens[0]
                attn_map = MultimodalAttentionVisualizer.generate_spatial_attention_map(
                    img_to_process,
                    token=key_token,
                    grid_size=8,
                )
                st.dataframe(attn_map, use_container_width=True)
