# Multimodal Content Intelligence Suite

[![CI](https://github.com/samaya8/omni-content-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/samaya8/omni-content-intelligence/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-45%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `omni-content-intelligence`  
> **Repository Description**: Enterprise-grade dual-modal intelligence suite featuring Graph & Neural Text Summarization, Vision-Language Image Captioning with Beam Search, Explainable Cross-Attention Heatmaps, and Pure-Python NLG Benchmark Metrics (ROUGE, BLEU, CIDEr).

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/pytest-45%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](https://coverage.readthedocs.io/)
[![Zero-GPU](https://img.shields.io/badge/zero--gpu-100%25%20CPU%20compatible-success.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

---

## 🌟 Executive Overview

**Multimodal Content Intelligence** is an enterprise-grade platform unifying natural language summarization and vision-language captioning under a single robust, explainable framework. Designed for production environments requiring deterministic execution, zero-GPU edge fallbacks, and verifiable NLG evaluation.

### Key Capabilities:
1. **Dual-Strategy Text Summarization**:
   - **Extractive Engine**: Graph-based **TextRank** with eigenvector centrality and **Maximal Marginal Relevance (MMR, $\lambda=0.7$)** redundancy suppression.
   - **Abstractive Pipeline**: Hugging Face Transformer integration (`facebook/bart-large-cnn`) backed by a deterministic neural synthesizer fallback for instant offline execution.
   - **Presentation Styles**: `executive` narrative paragraphs, `bullet_points` action items, and `dense` technical takeaway digests.
2. **Vision-Language Image Captioning**:
   - **Multi-Strategy Decoding**: Length-normalized **Beam Search ($K$-beams, $\alpha=0.7$)** vs **Greedy Search** with repetition penalization.
   - **Visual Concept Recognition**: Categorizes detected objects, attributes, actions, and environmental context.
   - **Zero-GPU Determinism**: High-fidelity perceptual hash simulation enabling 100% reproducible testing on edge hardware.
3. **Explainable Cross-Attention Heatmaps**:
   - **Spatial Visual Maps**: $8\times 8$ 2D Gaussian attention density distribution mapped per generated caption token.
   - **Text Salience Visualization**: Dynamic HTML span rendering coloring source document sentences by PageRank centrality.
4. **Standard NLG Evaluation Metrics**:
   - **ROUGE-1, ROUGE-2, ROUGE-L**: Precision, Recall, and F1 with dynamic programming Longest Common Subsequence (LCS).
   - **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: Modified n-gram precision with candidate-to-reference Brevity Penalty ($BP$).
   - **CIDEr Consensus**: TF-IDF weighted n-gram cosine similarities for multi-reference image descriptions.
   - **Abstractiveness & Compression**: Novel unigram/bigram novelty ratio and compression savings percentage.
5. **Dual Streamlit Applications**:
   - `app.py`: Clean operational studio for document summarization and image captioning.
   - `app2.py`: Multimodal Intelligence Lab with side-by-side Summarization Arena, Beam Visualizer, and interactive NLG benchmark calculator.
6. **Enterprise CLI & Batch Queue**: Full automation via `cli.py` and `evals/runner.py`.

---

## 📐 System Architecture

```text
                               ┌────────────────────────────────────────┐
                               │       MULTIMODAL INPUT SOURCE          │
                               └──────────────────┬─────────────────────┘
                                                  │
                    ┌─────────────────────────────┴─────────────────────────────┐
                    ▼                                                           ▼
         [Long-Form Text Document]                                       [Input Image File]
                    │                                                           │
        ┌───────────┴───────────┐                                   ┌───────────┴───────────┐
        ▼                       ▼                                   ▼                       ▼
 [Graph TextRank]       [Neural Synthesizer]               [Beam Search (K=3)]      [Greedy Search]
  - Sentence Graph       - Transformer / BART               - Length Penalty         - Fast Decoder
  - PageRank Iteration   - Paraphrasing                     - Candidate Beams        - Repetition Penalty
  - MMR Redundancy       - Narrative Linking                        │                       │
        │                       │                                   └───────────┬───────────┘
        └───────────┬───────────┘                                               ▼
                    ▼                                             [Caption & Detected Entities]
        [Distilled Text Summary]                                                │
                    │                                                           ▼
                    ▼                                             [Spatial 8x8 Attention Map]
       [Text Salience HTML View]                                                │
                    │                                                           │
                    └─────────────────────────────┬─────────────────────────────┘
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   PURE-PYTHON NLG EVAL SUITE    │
                                 │   - ROUGE-1 / ROUGE-2 / ROUGE-L │
                                 │   - BLEU-1 / 2 / 3 / 4 (BP)     │
                                 │   - CIDEr Consensus Metric      │
                                 │   - Novelty & Compression %     │
                                 └─────────────────────────────────┘
```

---

## 🚀 Quickstart & CLI Usage

The platform provides an enterprise command-line interface (`cli.py`) with 4 core subcommands:

### 1. Text Summarization
```bash
# Extractive summary with bullet point formatting
python cli.py summarize --text "Artificial intelligence transforms enterprise workflows. Transformers use self-attention to scale across distributed clusters." --strategy extractive --style bullet_points

# Abstractive summary from file with JSON output
python cli.py summarize --file report.txt --strategy abstractive --json
```

### 2. Image Captioning
```bash
# Generate caption with Beam Search (K=3) and concept entity tags
python cli.py caption --image photo.jpg --method beam --beam-size 3 --concepts

# Greedy decoding with plain stdout report
python cli.py caption --image photo.jpg --method greedy
```

### 3. NLG Benchmark Evaluation
```bash
# Evaluate candidate against reference with ROUGE, BLEU, and CIDEr
python cli.py evaluate \
  --candidate "A black dog runs through the grassy field." \
  --reference "A dark dog running across the green lawn." \
  --reference "A playful dog sprints outdoors in the grass." \
  --metric all \
  --json
```

### 4. Bulk Batch Processing
```bash
python cli.py batch --input tasks.json --output results.json
```

---

## 🖥️ Streamlit Applications

### 1. Studio Application (`app.py`)
User-friendly workspace for content distillation, text salience highlighting, and visual scene description:
```bash
streamlit run app.py
```

### 2. Lab & Arena Workbench (`app2.py`)
Diagnostic workbench for head-to-head model ablation, beam search tree inspection, spatial cross-attention heatmaps, and live NLG calculations:
```bash
streamlit run app2.py
```

---

## 📊 Benchmark Verification

Automated evaluation over golden enterprise benchmarks is run via `evals/runner.py`:
```bash
python evals/runner.py
```

### Summary of Benchmark Results:
| Strategy / Method | Key Metric 1 | Key Metric 2 | Compression / Alignment | Avg Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: |
| **Extractive Summarizer** | ROUGE-1: **34.9%** | ROUGE-L: **30.1%** | **49.2%** token savings | `< 1.0 ms` |
| **Abstractive Summarizer** | ROUGE-1: **39.1%** | ROUGE-L: **31.3%** | **5.4%** novel unigrams | `< 1.0 ms` |
| **Beam Captioner (K=3)** | BLEU-1: **43.8%** | BLEU-4: **5.3%** | CIDEr: **0.42** | `< 1.0 ms` |
| **Greedy Captioner (K=1)** | BLEU-1: **43.8%** | BLEU-4: **5.3%** | CIDEr: **0.42** | `< 1.0 ms` |

*Full audit report available in [`evals/BENCHMARK_RESULTS.md`](evals/BENCHMARK_RESULTS.md).*

---

## 🧪 Testing & Code Coverage

Run the complete test suite with line-by-line coverage analysis:
```bash
pytest --cov=src --cov=cli --cov=evals --cov-report=term-missing tests/
```

```text
Name                  Stmts   Miss  Cover   Missing
---------------------------------------------------
cli.py                  219     23    89%   
evals\runner.py          77      5    94%   
src\attention.py         93      6    94%   
src\captioner.py         85      7    92%   
src\eval_metrics.py     137      2    99%   
src\summarizer.py       137     13    91%   
src\text_utils.py        23      2    91%   
---------------------------------------------------
TOTAL                   771     58    92%
============================= 45 passed in 0.55s ==============================
```

---

## 📁 Repository Structure

```text
.
├── app.py                          # Streamlit Content Intelligence Studio
├── app2.py                         # Streamlit Lab, Arena & Benchmark Calculator
├── cli.py                          # Enterprise CLI (summarize, caption, evaluate, batch)
├── evals/
│   ├── runner.py                   # Automated benchmark evaluation runner
│   └── BENCHMARK_RESULTS.md        # Comprehensive benchmark report
├── notebooks/
│   ├── 03- text-summarization-using-lstm.ipynb           # Seq2Seq LSTM text summarization
│   ├── 03- text-summarization-using-bert-hf.ipynb        # Hugging Face BART transformer
│   └── 03- image-caption-generator-using-cnn-lstm.ipynb # CNN-LSTM image captioning
├── src/
│   ├── __init__.py                 # Package root
│   ├── attention.py                # Spatial 2D attention maps & HTML salience renderer
│   ├── captioner.py                # Vision-language beam search & entity recognizer
│   ├── eval_metrics.py             # Pure-Python ROUGE-1/2/L, BLEU-1..4, CIDEr, Novelty
│   ├── summarizer.py               # Graph TextRank + MMR & Neural Abstractive Synthesizer
│   └── text_utils.py               # Legacy text tokenization and normalization
├── tests/
│   ├── conftest.py                 # Pytest environment configuration
│   ├── test_attention.py           # Attention & salience tests (6 passed)
│   ├── test_captioner.py           # Captioning & beam search tests (6 passed)
│   ├── test_cli.py                 # CLI end-to-end command tests (12 passed)
│   ├── test_eval_metrics.py        # ROUGE, BLEU, CIDEr tests (7 passed)
│   ├── test_evals.py               # Benchmark runner tests (3 passed)
│   ├── test_summarizer.py          # Summarizer & MMR tests (8 passed)
│   └── test_text_utils.py          # Text utils legacy tests (3 passed)
├── requirements.txt                # Pinned dependencies
├── LICENSE                         # MIT License
├── PROJECT_DETAILS.md              # Technical specification & mathematical formulations
└── README.md                       # Repository documentation
```

---

## 📜 License & Citation

Released under the **MIT License**.
If you use this software in research or production deployments, please cite:
```bibtex
@software{multimodal_content_intelligence_2026,
  author = {AI & Machine Learning Engineering Team},
  title = {Multimodal Content Intelligence: Dual-Strategy Summarization, Vision Captioning, and NLG Evaluation},
  year = {2026}
}
```
