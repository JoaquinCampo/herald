# Section 2: Catastrophic failures under KV-cache compression (prose draft)

## 2.1 What we mean by *catastrophic*

We distinguish two regimes of LLM failure under KV-cache compression. The first is graceful accuracy degradation: the model finishes generating, returns a parseable answer, and is simply wrong. This regime is captured by standard task-accuracy benchmarks and tracked by every published evaluation of KV-cache compression. The second regime is qualitatively different. The generation itself goes wrong: the model emits the same fragment over and over, runs to the token budget without ever producing a stop token, or both. We call this regime *catastrophic*. It is not a measure of how good the answer is; it is a measure of whether the model produced a usable answer at all.

We formalize two catastrophic failure modes:

- **Looping**, the model emits a chunk of tokens, then re-emits the same chunk, then re-emits it again, with no new content in between.
- **Non-termination**, the model never produces an end-of-sequence token within the generation budget.

We do not include three plausible candidates: *wrong-answer* (a normal mode of LLM failure even on uncompressed inference, see §2.3), *instruction amnesia* (the model solves a different problem than asked), and *coherence collapse* (the output becomes ungrammatical mid-stream). Each is real and occasionally observed in our data, but none admits a robust automatic detector at our scale, and we leave them as future work.

Looping and non-termination earn their place because they are (i) detectable from the token stream alone, with no judge model required; (ii) demonstrably caused by compression, their rates rise sharply with compression aggressiveness and are near zero on uncompressed runs (§2.3); and (iii) operationally severe, since both produce unusable outputs and both run the model to its full token budget.

## 2.2 Detection rules

\paragraph{Non-termination.} We mark a generation as non-terminating if the decoding loop exited because it hit the token budget rather than because the model produced an end-of-sequence token, i.e. its stop reason is one of \texttt{max\_tokens} or \texttt{timeout}. Non-termination is a property of the entire trace, not of a moment within it; for the hazard labels of §3 we treat the catastrophe onset as the final emitted token.

\paragraph{Looping.} We mark a generation as looping if any window of $W = 20$ contiguous tokens appears at least $r = 3$ times within the generated sequence. The onset of the loop is the start position of the second occurrence of the first window that hits the threshold, the first token at which the model committed to repeating itself instead of producing new content. The thresholds are calibrated to be tight enough to catch short phrase-level loops (the typical KV-cache catastrophe arc, see Figure~\ref{fig:teaser}, left) and loose enough to ignore natural repetition such as enumerated lists, where short windows can recur without indicating failure. Setting $r = 3$ rather than $r = 2$ is what makes the rule robust to coincidental pairs.

\paragraph{Co-occurrence.} The two modes are not independent. A looping generation almost always escalates into non-termination, since once the model commits to a loop it has no mechanism to break out and exhausts the token budget. We report the two detectors separately because they have different temporal signatures (looping has a clear within-trace onset; non-termination does not), but in §3 we treat the union of either firing as a single positive hazard label.

\paragraph{Wrong-answer carve-out.} We measure final-answer correctness alongside, but we do not treat wrong answers as a hazard label. The reason is empirical: even with no compression at all, Qwen2.5-7B-Instruct produces a $22.4\%$ wrong-answer rate on GSM8K ($112 / 500$ sequences in the uncompressed baseline, with task accuracy $77.6\%$). Including wrong answers in the hazard target would inject a base rate of label noise unrelated to compression and would force HERALD to predict a behavior the model exhibits even when nothing is wrong with its cache. We track wrong-answer rate as an accuracy metric in §6 and otherwise keep it out of the catastrophic label.

## 2.3 Prevalence across compressors

\todo{Bar chart figure: catastrophic rate per compressor, ordered worst to best, with the uncompressed baseline as a horizontal reference line.}

Table~\ref{tab:per-press} reports the fraction of sequences exhibiting at least one catastrophe (looping or non-termination) under each compressor in our sweep. Numbers are sequence-level: a sequence counts as catastrophic if either detector fires anywhere in its trace.

\begin{table}[h]
\centering
\small
\begin{tabular}{lrr}
\toprule
\textbf{Compressor} & \textbf{Catastrophic rate} & $\bm{n_{\text{cat}} / n_{\text{seq}}}$ \\
\midrule
KNorm              & 71.0\% & 1774 / 2500 \\
SnapKV             & 68.7\% & 1690 / 2460 \\
Random             & 35.9\% & 898 / 2500 \\
TOVA               & 15.4\% & 385 / 2500 \\
StreamingLLM       & 13.7\% & 342 / 2500 \\
ExpectedAttention  & 4.8\%  & 121 / 2500 \\
\midrule
\textit{None (uncompressed)} & \textit{1.2\%} & \textit{6 / 500} \\
\bottomrule
\end{tabular}
\caption{Sequence-level catastrophic rate per compressor on GSM8K with Qwen2.5-7B-Instruct, pooled across all evaluated compression ratios. The uncompressed baseline establishes the model's intrinsic catastrophe rate on this task.}
\label{tab:per-press}
\end{table}

Three observations stand out. First, catastrophic behavior is highly compressor-specific: there is a roughly $15\times$ spread between the most fragile compressor (KNorm at 71\%) and the most robust (ExpectedAttention at 4.8\%) on the same prompts and the same set of compression ratios. Second, two of the four "principled" compressors in our sweep (KNorm and SnapKV) are *worse than random eviction* in this regime, not by a small margin, but by roughly a factor of two. Random eviction (35.9\%) sits below both, which is striking given that random is meant as a sanity baseline rather than a competitive method. Third, the uncompressed baseline is non-zero (1.2\%): a small fraction of long-form chain-of-thought generations on GSM8K loop or fail to terminate even with the full cache available. Any predictor of compression-induced catastrophes therefore has to clear an intrinsic floor; the true effect we want to model is the gap between this floor and the per-compressor rates above it.

## 2.4 Prevalence across compression ratios

\todo{Line plot figure: catastrophic rate vs. compression ratio, one curve per compressor, with the uncompressed baseline as a horizontal reference line.}

Pooling all compressors at each ratio gives a clean view of how catastrophic behavior scales with compression aggressiveness (Table~\ref{tab:per-ratio}).

\begin{table}[h]
\centering
\small
\begin{tabular}{rr}
\toprule
\textbf{Compression ratio} & \textbf{Catastrophic rate} \\
\midrule
0.000 (uncompressed) & 1.2\% \\
0.250 & 10.1\% \\
0.500 & 25.5\% \\
0.625 & 35.9\% \\
0.750 & 45.9\% \\
0.875 & 56.7\% \\
\bottomrule
\end{tabular}
\caption{Sequence-level catastrophic rate vs. compression ratio, pooled across all six compressors evaluated at each ratio. Each $0.125$ step in compression adds roughly ten percentage points of catastrophic behavior.}
\label{tab:per-ratio}
\end{table}

The relationship is approximately linear: each $0.125$ step in compression adds roughly ten percentage points of catastrophic behavior, all the way from the floor at zero compression up to $56.7\%$ at the heaviest ratio we evaluate. There is no obvious "safe" heavy regime. By ratio $0.875$, more than half of all generations across all six compressors fail catastrophically, which inverts the usual mental model in which catastrophes are tail events to be ignored when reporting average accuracy: at heavy ratios they are the modal outcome.

The pooled view also hides a useful asymmetry that Table~\ref{tab:per-press} surfaces. The headline rate at $0.875$ is dominated by the fragile compressors (KNorm and SnapKV); the robust ones (ExpectedAttention) stay tolerable even there. This is precisely the tension that motivates a runtime predictor: an inference system that wants the memory savings of heavy compression cannot trust an averaged per-compressor accuracy number, because the variance across compressors and across individual sequences is enormous and concentrates exactly where it hurts.

## 2.5 What a catastrophe looks like

Figure~\ref{fig:teaser} (left) tracks a single catastrophic generation under random eviction at ratio $0.75$. The model is asked a one-step GSM8K word problem, solves it correctly in the first part of the trace ($200$ kg total), then drifts into a spurious self-correction (``However, the question asks for…''), and finally collapses into four near-identical repetitions of the same boxed answer before being cut off at the token budget. The arc of correct math, then drift, then loop crystallization, then non-termination is the canonical compression-induced catastrophe shape; both detectors fire on this single trace. The hazard score that HERALD will produce in §3, drawn as the vertical bar in the figure, rises tens of tokens before any of the externally visible failure becomes apparent.
