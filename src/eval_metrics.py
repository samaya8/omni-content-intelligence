"""Natural Language Generation (NLG) Evaluation Metrics Suite.

Implements mathematically rigorous, pure-Python standards:
1. ROUGE-1, ROUGE-2, ROUGE-L (Recall, Precision, F1 with Longest Common Subsequence).
2. BLEU-1, BLEU-2, BLEU-3, BLEU-4 (with Modified N-gram Precision & Brevity Penalty).
3. CIDEr Consensus Metric (Consensus-based Image Description Evaluation via TF-IDF n-gram vectors).
4. Compression Ratio & Novel N-gram Abstractiveness Ratio.
"""

from __future__ import annotations

import collections
import math
import re
from typing import Any, Counter, Dict, List, Optional, Set, Tuple, Union


class NLGEvalMetrics:
    """Computes ROUGE, BLEU, CIDEr, and abstractiveness scores for text and captions."""

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Generate sequence of n-grams from token list."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def _longest_common_subsequence(seq_a: List[str], seq_b: List[str]) -> int:
        """Compute the length of the Longest Common Subsequence (LCS) using dynamic programming."""
        m, n = len(seq_a), len(seq_b)
        if m == 0 or n == 0:
            return 0

        # Memory-efficient 2-row DP table
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    @classmethod
    def compute_rouge(
        cls,
        candidate: str,
        reference: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1)."""
        cand_tokens = cls._tokenize(candidate)
        ref_tokens = cls._tokenize(reference)

        results: Dict[str, Dict[str, float]] = {}

        for n, name in [(1, "rouge-1"), (2, "rouge-2")]:
            cand_ngrams = collections.Counter(cls._get_ngrams(cand_tokens, n))
            ref_ngrams = collections.Counter(cls._get_ngrams(ref_tokens, n))

            total_cand = sum(cand_ngrams.values())
            total_ref = sum(ref_ngrams.values())

            if total_cand == 0 or total_ref == 0:
                results[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                continue

            overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())

            prec = overlap / total_cand
            rec = overlap / total_ref
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            results[name] = {
                "precision": round(prec * 100.0, 2),
                "recall": round(rec * 100.0, 2),
                "f1": round(f1 * 100.0, 2),
            }

        # ROUGE-L (LCS)
        lcs_len = cls._longest_common_subsequence(cand_tokens, ref_tokens)
        len_cand = len(cand_tokens)
        len_ref = len(ref_tokens)

        if len_cand == 0 or len_ref == 0:
            results["rouge-l"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            prec_l = lcs_len / len_cand
            rec_l = lcs_len / len_ref
            f1_l = (2 * prec_l * rec_l) / (prec_l + rec_l) if (prec_l + rec_l) > 0 else 0.0
            results["rouge-l"] = {
                "precision": round(prec_l * 100.0, 2),
                "recall": round(rec_l * 100.0, 2),
                "f1": round(f1_l * 100.0, 2),
            }

        return results

    @classmethod
    def compute_bleu(
        cls,
        candidate: str,
        references: Union[str, List[str]],
        max_n: int = 4,
    ) -> Dict[str, float]:
        """Compute BLEU-1 through BLEU-4 with modified n-gram precision and Brevity Penalty."""
        if isinstance(references, str):
            references = [references]

        cand_tokens = cls._tokenize(candidate)
        ref_tokens_list = [cls._tokenize(r) for r in references]

        c = len(cand_tokens)
        if c == 0:
            return {f"bleu-{i}": 0.0 for i in range(1, max_n + 1)} | {"brevity_penalty": 0.0}

        # Closest reference length for brevity penalty
        r = min((abs(len(ref) - c), len(ref)) for ref in ref_tokens_list)[1]

        bp = 1.0 if c > r else math.exp(1.0 - r / c)

        precisions: List[float] = []
        for n in range(1, max_n + 1):
            cand_ngrams = collections.Counter(cls._get_ngrams(cand_tokens, n))
            total_cand = sum(cand_ngrams.values())

            if total_cand == 0:
                precisions.append(0.0)
                continue

            # Maximum reference count for each n-gram
            max_ref_counts: Dict[Tuple[str, ...], int] = {}
            for ref_tokens in ref_tokens_list:
                ref_ngrams = collections.Counter(cls._get_ngrams(ref_tokens, n))
                for ng, count in ref_ngrams.items():
                    max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), count)

            clipped_matches = sum(min(count, max_ref_counts.get(ng, 0)) for ng, count in cand_ngrams.items())
            # Add-1 smoothing (Chen & Cherry) if no overlap to prevent log(0)
            smoothed_prec = (clipped_matches + 0.1) / (total_cand + 0.1) if clipped_matches == 0 else clipped_matches / total_cand
            precisions.append(smoothed_prec)

        results: Dict[str, float] = {"brevity_penalty": round(bp, 4)}
        for n in range(1, max_n + 1):
            log_avg = sum(math.log(p) for p in precisions[:n]) / n
            score = bp * math.exp(log_avg) * 100.0
            results[f"bleu-{n}"] = round(score, 2)

        return results

    @classmethod
    def compute_cider(
        cls,
        candidate: str,
        references: List[str],
        n_max: int = 4,
    ) -> float:
        """Compute CIDEr consensus score for image captioning against reference descriptions."""
        if not references or not candidate.strip():
            return 0.0

        cand_tokens = cls._tokenize(candidate)
        ref_tokens_list = [cls._tokenize(r) for r in references]
        m = len(ref_tokens_list)

        # Build corpus of reference documents for IDF weights
        all_docs = ref_tokens_list + [cand_tokens]
        num_docs = len(all_docs)

        cider_n_scores = []
        for n in range(1, n_max + 1):
            cand_ngrams = collections.Counter(cls._get_ngrams(cand_tokens, n))
            ref_ngrams_list = [collections.Counter(cls._get_ngrams(r, n)) for r in ref_tokens_list]

            # Document frequencies for n-grams
            df_map: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
            for doc_ngrams in [cand_ngrams] + ref_ngrams_list:
                for ng in doc_ngrams.keys():
                    df_map[ng] += 1

            # TF-IDF vectors
            def to_tfidf(ng_counter: Counter) -> Dict[Tuple[str, ...], float]:
                vec = {}
                tot = sum(ng_counter.values()) or 1
                for ng, cnt in ng_counter.items():
                    tf = cnt / tot
                    idf = math.log((num_docs + 1) / (df_map[ng] + 0.5)) + 1.0
                    vec[ng] = tf * idf
                return vec

            cand_vec = to_tfidf(cand_ngrams)
            cand_norm = math.sqrt(sum(v * v for v in cand_vec.values())) or 1e-6

            cos_sims = []
            for ref_ngrams in ref_ngrams_list:
                ref_vec = to_tfidf(ref_ngrams)
                ref_norm = math.sqrt(sum(v * v for v in ref_vec.values())) or 1e-6

                dot = sum(cand_vec.get(ng, 0.0) * ref_vec.get(ng, 0.0) for ng in cand_vec)
                cos_sims.append(dot / (cand_norm * ref_norm))

            cider_n_scores.append(sum(cos_sims) / m)

        # Average across n-grams and scale to 0-10 benchmark scale
        return round(float(sum(cider_n_scores) / n_max * 10.0), 2)

    @classmethod
    def compute_abstractiveness_and_compression(
        cls,
        candidate: str,
        source: str,
    ) -> Dict[str, float]:
        """Measure compression ratio and novel n-gram abstractiveness ratio."""
        cand_tokens = cls._tokenize(candidate)
        src_tokens = cls._tokenize(source)

        cand_len = len(cand_tokens)
        src_len = len(src_tokens)

        comp_ratio = round(cand_len / max(1, src_len), 3)

        # Novel unigrams & bigrams
        src_unigrams = set(src_tokens)
        src_bigrams = set(cls._get_ngrams(src_tokens, 2))

        cand_unigrams = cls._get_ngrams(cand_tokens, 1)
        cand_bigrams = cls._get_ngrams(cand_tokens, 2)

        novel_1 = sum(1 for (u,) in cand_unigrams if u not in src_unigrams) / max(1, len(cand_unigrams))
        novel_2 = sum(1 for b in cand_bigrams if b not in src_bigrams) / max(1, len(cand_bigrams))

        return {
            "compression_ratio": comp_ratio,
            "compression_savings_pct": round((1.0 - comp_ratio) * 100.0, 1),
            "novel_unigram_ratio": round(novel_1 * 100.0, 1),
            "novel_bigram_ratio": round(novel_2 * 100.0, 1),
        }
