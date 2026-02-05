---
title: "Benchmarking Agent Memory Architectures: Which RAG Strategy Actually Works?"
short_description: "How we tested 10 memory strategies across 4 LLMs and 5,800+ evaluations to find what works for production agent systems"
description: "How we tested 10 memory strategies across 4 LLMs and 5,800+ evaluations to find what works for production agent systems"
draft: false
---

How we tested 10 memory strategies across 4 LLMs and 5,800+ evaluations to find what works for production agent systems

---

## The Problem

LLM agents have a memory problem.

Retrieval-Augmented Generation (RAG) was supposed to solve it: store your data in a vector database, retrieve relevant chunks at query time, feed them to the LLM. Simple. But as agent systems get more complex (multi-turn conversations, tool use chains, cross-domain reasoning), "just retrieve and stuff" breaks down in specific, measurable ways.

We've observed this directly from customer deployments:

- **Context noise**: Retrieving 15 chunks when only 3 are relevant dilutes the LLM's attention. Faithfulness drops from 0.98 to 0.85 as irrelevant context increases.
- **Context rot over turns**: In multi-turn conversations, agents lose track of facts discussed 5+ turns ago. Accuracy degrades as conversation history grows.
- **Latency kills engagement**: If latency goes beyond 1.5 seconds, conversations extending more than three or four turns get cut in half. For e-commerce, one second of waiting takes around 77% of the chance someone is buying.
- **One size doesn't fit all**: A cost-sensitive high-volume chatbot has different needs than an internal IT support agent. There is no single best architecture.

Our own earlier work on [skill.md meets REPL](https://qdrant.tech/articles/skill-md-meets-repl/) identified two orthogonal failure modes when agents interact with APIs: **known unknowns** (agents lack domain knowledge about best practices) and **unknown unknowns** (agents can't discover environment-specific resources). The solution was two complementary context layers: a static skill file with decision tables and gotchas, plus a dynamic REPL for runtime state discovery. Neither alone is sufficient: skill.md without REPL writes syntactically correct code that fails at runtime, while REPL without skill.md still uses deprecated methods.

That insight, that agents need *layered* context rather than just more context, is the foundation of this benchmark. We extend it with ideas from several directions:

- **Passive context injection**: Always-present system prompt knowledge beats on-demand retrieval. The agent shouldn't have to "decide" to use knowledge. It should always be there.
- **Sub-LLM compression**: Don't dump raw retrieved context into the main LLM's window. Use a smaller model to compress/summarize first, keeping the main model's context lean and focused.
- **Incremental folding**: Build summaries as interactions happen, rather than compressing at query time. Retain full context within a branch, but only a self-chosen summary persists.
- **Hierarchical summarization**: Every retrieval action produces both results and a summary. Summaries consolidate into higher-level insights over time.
- **Self-improving memory**: A reflector agent evaluates each interaction, extracts lessons, and injects them into future retrievals. The system gets better with use.

We built 10 distinct memory strategies from these ideas. Then we benchmarked them.

This article shares our results across 5,800+ evaluations, 4 LLMs, and 5 conversation scenarios. The goal: give you a decision framework for choosing the right memory architecture for your use case, backed by data.

---

## The Architectures

All strategies share the same foundation: Qdrant hybrid search (dense + sparse vectors with BM25) combined with Neo4j knowledge graph expansion and PostgreSQL conversation history. They differ in what happens between retrieval and generation.

### Tier 1: Baseline Strategies

**`full`: Full System Retrieval**

The baseline. Qdrant hybrid search + Neo4j graph expansion + PostgreSQL conversation history. All retrieved chunks are concatenated and passed directly to the LLM. No processing, no compression.

```
Query → [Qdrant Hybrid + Neo4j Graph + PG History] → Raw Chunks → LLM → Answer
```

**`skill_progressive`: Progressive Disclosure**

Separate what the agent *should know* from what it *needs to discover*. Instead of retrieving everything, first classify the query type (product info, refund policy, account help, etc.) using a decision table, then retrieve only from relevant sources. The classification step acts as the "known unknowns" layer, while targeted retrieval handles the "unknown unknowns."

```
Query → [Haiku: Classify] → [Targeted Retrieval] → Focused Chunks → LLM → Answer
```

### Tier 2: Context Engineering Strategies

**`context_pack`: Passive Context Packing**

Assembles a persistent knowledge index that's always injected into the system prompt. Includes: user's knowledge graph triples (pipe-delimited), compressed conversation summary, and top document chunks. The LLM doesn't have to "decide" to retrieve because knowledge is always present.

```
Query → [Qdrant + Neo4j + PG] → Raw Chunks + [System: Knowledge Index] → LLM → Answer
```

**`rlm_compress`: Strict RLM Compression**

Full retrieval, then a sub-LLM (Haiku) extracts only directly relevant facts. Aggressive: "extract ONLY the facts... Omit anything irrelevant." Max 300 tokens output.

```
Query → [Full Retrieval] → Raw Chunks → [Haiku: Extract ONLY relevant] → Compressed → LLM → Answer
```

**`rlm_soft`: Soft RLM Compression**

Same pipeline, but the sub-LLM reorganizes and deduplicates instead of stripping. "Keep ALL facts that could be even partially relevant." Max 600 tokens output. Designed to fix the faithfulness degradation we observed with strict compression.

```
Query → [Full Retrieval] → Raw Chunks → [Haiku: Reorganize, deduplicate] → Organized → LLM → Answer
```

### Tier 3: Stateful Strategies

**`fold`: Context Folding**

Instead of compressing at query time, builds an incremental fold. Each retrieval merges new information into an evolving summary stored per-user. At query time, the fold replaces raw chunks. Nothing is discarded, and new facts integrate with existing knowledge.

```
Query → [Full Retrieval] → New Chunks + [Load Existing Fold] → [Haiku: Merge] → Updated Fold → LLM → Answer
                                                                                    ↓
                                                                              [Save Fold]
```

**`agentfold`: Hierarchical Action Summaries**

Every retrieval produces both results and a Level-0 summary. After 10 actions, L0 summaries consolidate into an L1 summary. Context includes raw results + hierarchical memory. The knowledge index grows more structured over time.

```
Query → [Full Retrieval] → Raw Chunks → [Haiku: Summarize Action] → L0 Summary
                               ↓                                        ↓
                          Raw + L1 Summaries + Recent L0s           [Store L0]
                               ↓                                   [Consolidate → L1 if needed]
                              LLM → Answer
```

**`reflector`: Self-Improving Memory**

After each answer, a Reflector agent evaluates the interaction: what was learned, what was missing, how good was the answer. Lessons accumulate and are injected into future retrievals. The system literally gets better with use.

```
Query → [Full Retrieval] → Raw Chunks + [Prior Lessons] → LLM → Answer
                                                              ↓
                                                    [Reflector: What did we learn?]
                                                              ↓
                                                       [Store Lessons]
```

### Combined Strategies

**`skill_progressive_pack`**: Progressive disclosure + passive context pack. Best recall.

**`context_pack_rlm`**: Passive context + strict compression. Tests if both together outperform either alone.

---

## Experimental Setup

### Dataset

**Single-shot evaluation**: 100 questions across 5 types, testing distinct capabilities:

| Type | Count | Tests |
|------|-------|-------|
| Single-hop | 30 | Direct fact retrieval ("Where is Qdrant headquartered?") |
| Multi-hop | 24 | Cross-referencing multiple facts ("What deployment options does the company Daniel works at offer?") |
| Temporal | 15 | Conversation history awareness ("What did we discuss about refund policies?") |
| Recall | 15 | Informal reference resolution ("What was that thing about the 30 days?") |
| Adversarial | 16 | Correcting false premises ("Is Qdrant headquartered in Munich?") |

**Multi-turn scenarios**: 5 conversation scenarios with 38 total turns, testing context retention over realistic interactions:

| Scenario | Turns | Tests |
|----------|-------|-------|
| Customer Onboarding | 6 | Progressive learning, back-references, full summary |
| Refund Escalation | 7 | Temporal reasoning, counterfactuals, context switching |
| Cross-Domain Reasoning | 5 | Entity lookup, cross-domain synthesis |
| Adversarial Conversation | 5 | Correction of false premises, recall after adversarial turns |
| Long Horizon | 15 | 10 facts introduced, then recall of early facts at turn 11-15 |

All questions are grounded in two knowledge domains stored in Qdrant: **Qdrant company facts** (headquarters, employees, features, certifications, deployments) and a **refund policy** (timelines, processing, premium benefits). This mirrors a real customer support knowledge base.

### Models

| Model | Provider | Strengths |
|-------|----------|-----------|
| Claude Sonnet | Anthropic | Highest quality, best for complex reasoning |
| Claude Haiku | Anthropic | Fast, cheap, surprisingly high faithfulness |
| GPT-4o | OpenAI | Fastest total response time |
| GPT-4o-mini | OpenAI | Cheapest per eval, good speed |

All models run at temperature=0 for reproducibility. Metrics are scored by dual blind judges: Claude Haiku and GPT-4o-mini score independently, and the final score is the average. This reduces single-model bias.

### Metrics

**Quality metrics** (0-1, higher is better):
- **Faithfulness**: Does the answer only contain information from the provided context?
- **Context Recall**: Did the retrieval find all the relevant information?
- **Context Precision**: Is the retrieved context mostly relevant (not noisy)?
- **Answer Relevancy**: Does the answer actually address the question?

**Production metrics**:
- **Total latency**: End-to-end response time (retrieval + processing + generation)
- **Retrieval latency**: Time spent on retrieval only (what we control)
- **Cost per eval**: Token costs across all LLM calls (main + sub-LLMs)
- **P50/P95/P99 latencies**: Percentile distributions for SLA compliance

### Latency SLA Targets

Derived from customer deployment requirements:

| Target | Threshold | Source |
|--------|-----------|--------|
| Engagement/Ambassador | < 1.5s | "If latency goes beyond 1.5s, multi-turn conversations get cut in half" |
| E-commerce | < 2s | "I want one second or max two seconds response time" |
| Agent Streaming | < 3.5s | "3.5 seconds is the max the AI agent will take to start streaming" |
| IT Support | < 5s | "Between three to four seconds is where I would see it" |
| Unacceptable | > 5s | "One customer had 5 seconds response time and that's slow" |

### Statistical Methodology

Raw averages can mislead. A strategy scoring 0.95 faithfulness on 100 questions might have a 95% confidence interval of [0.92, 0.98], meaning the true performance could be anywhere in that range. We address this with:

- **Bootstrap confidence intervals** (10,000 resamples) for all quality metrics
- **Paired Wilcoxon signed-rank tests** between strategies on the same questions, with Bonferroni correction
- **Cohen's d effect sizes** to distinguish statistical from practical significance
- **Inter-rater reliability**: Cohen's kappa and Pearson correlation between our two judges
- **pass@k consistency**: Running k=3 trials at temperature=0 to measure non-determinism from HNSW approximate nearest neighbors, embedding variance, and Neo4j path ordering
- **pass^k reliability**: Probability of ALL k trials succeeding, which is a stricter measure of consistency. A 75% per-trial success rate over 3 trials yields just 42% pass^3.
- **Eval saturation detection**: Flagging metrics where scores are near-ceiling (mean >= 0.95, 50%+ perfect), meaning the eval can no longer discriminate improvements
- **Hallucination resistance**: 10 unanswerable questions (information genuinely absent from the KB) scored on whether the agent says "I don't know" or fabricates an answer
- **Full transcript saving**: Every question/answer/context triple saved for manual review, since automated graders can't catch everything

All confidence intervals reported as mean ± margin (95% CI). Statistical significance at p < 0.05.

### Token & Cost Efficiency

Beyond raw quality scores, we measure:

- **Token efficiency**: faithfulness per 1,000 tokens consumed (normalized against `full` baseline)
- **Cost efficiency**: faithfulness per dollar spent
- **Latency efficiency**: faithfulness per second of wall time
- **Pareto frontier**: strategies that are not dominated on (quality, cost). If another strategy is both higher quality AND cheaper, the first is dominated

### Component Ablation

To isolate the marginal contribution of each retrieval component, we run a systematic ablation:

1. Dense vectors only (Qdrant)
2. Sparse/BM25 only
3. Hybrid (dense + sparse)
4. \+ Neo4j graph expansion
5. \+ PostgreSQL conversation history (= `full` baseline)
6. \+ Passive context packing
7. \+ Soft compression
8. \+ Everything combined

Each transition adds exactly one component. We report the delta in faithfulness and latency, with statistical significance.

### Scale Testing

Production deployments have 10K-1M+ documents, not 100. We benchmark the same 100 eval questions against collections of increasing size to measure:

- Retrieval latency degradation as index grows
- Context precision drop (more distractors = more noise)
- Whether compression strategies help more at scale (hypothesis: yes)

---

## Results: Single-Shot Evaluation

### Overall Performance (110 questions: 100 answerable + 10 unanswerable, temperature=0)

#### Claude Sonnet, Answerable Questions (averaged across 3 independent trials, n=300 per config)

| Strategy | Faithful | Recall | Precision | Relevancy | Total P50 (ms) | TTFT P50 (ms) | $/eval | Verdict |
|----------|----------|--------|-----------|-----------|----------------|---------------|--------|---------|
| `full` | 0.946 | 0.776 | 0.872 | 0.856 | 3,501 | 983 | $0.006 | ACCEPTABLE |
| `context_pack` | **0.967** | 0.784 | 0.871 | **0.878** | 10,956 | 6,374 | $0.008 | TOO SLOW |
| `rlm_soft` | 0.937 | **0.814** | **0.878** | 0.838 | 7,204 | 969 | $0.003 | TOO SLOW |
| `skill_progressive` | 0.911 | **0.810** | 0.854 | 0.865 | 4,034 | 987 | $0.004 | ACCEPTABLE |

Additional strategies from single-trial runs:

| Strategy | Faithful | Recall | Precision | Relevancy | Total P50 (ms) | $/eval | Verdict |
|----------|----------|--------|-----------|-----------|----------------|--------|---------|
| `rlm_compress` | 0.935 | 0.879 | **0.934** | 0.803 | 4,600 | $0.002 | ACCEPTABLE |
| `skill_progressive_pack` | 0.921 | **0.949** | 0.890 | **0.877** | 5,500 | $0.005 | TOO SLOW |
| `context_pack_rlm` | 0.845 | 0.882 | **0.942** | 0.858 | 12,000 | $0.004 | TOO SLOW |
| `fold` | 0.951 | **0.946** | 0.855 | 0.840 | 10,918 | $0.004 | TOO SLOW |
| `agentfold` | 0.950 | 0.916 | 0.913 | **0.876** | 6,487 | $0.009 | TOO SLOW |
| `reflector` | 0.951 | 0.916 | 0.912 | 0.863 | 3,613 | $0.006 | ACCEPTABLE |

#### Claude Sonnet, Unanswerable Questions (averaged across 3 trials, n=30 per config)

| Strategy | Faithfulness | Abstention Score | N |
|----------|-------------|-----------------|---|
| `full` | 0.925 | **1.000** | 30 |
| `context_pack` | 0.913 | **1.000** | 27 |
| `rlm_soft` | 0.813 | **1.000** | 30 |
| `skill_progressive` | 0.728 | **1.000** | 30 |

All strategies achieve perfect abstention (correctly say "I don't know"), but `skill_progressive` shows significantly lower faithfulness on unanswerable questions (0.728). The classification step sometimes routes to irrelevant sources that produce partially confident responses.

> **Reproducibility note**: Per-trial faithfulness varies by ±0.01 across 3 independent trials (e.g., `full`: T1=0.944, T2=0.942, T3=0.952). This confirms that single-run benchmarks are reasonably stable for aggregate metrics, but individual questions can vary significantly (see pass^k analysis below).

#### Claude Haiku

| Strategy | Faithful | Recall | Precision | Relevancy | Total (ms) | $/eval |
|----------|----------|--------|-----------|-----------|------------|--------|
| `full` | **0.985** | 0.911 | 0.912 | 0.826 | 3,159 | $0.003 |
| `context_pack` | 0.969 | 0.917 | 0.910 | **0.867** | 6,144 | $0.003 |
| `rlm_compress` | 0.939 | 0.893 | **0.941** | 0.826 | 4,400 | **$0.001** |
| `skill_progressive` | 0.934 | **0.947** | 0.890 | 0.829 | 3,923 | $0.002 |
| `skill_progressive_pack` | 0.907 | **0.947** | 0.892 | 0.839 | 6,500 | $0.002 |
| `context_pack_rlm` | 0.917 | 0.892 | **0.941** | 0.829 | 7,775 | $0.001 |
| `rlm_soft` | 0.974 | **0.957** | **0.964** | 0.826 | 6,690 | $0.001 |
| `fold` | 0.925 | 0.938 | 0.838 | 0.815 | 9,879 | $0.002 |
| `agentfold` | 0.969 | 0.913 | 0.915 | 0.835 | 5,867 | $0.004 |
| `reflector` | **0.982** | 0.918 | 0.914 | 0.832 | 2,756 | $0.003 |

#### GPT-4o (100 questions)

| Strategy | Faithful | Recall | Precision | Relevancy | Total (ms) | $/eval |
|----------|----------|--------|-----------|-----------|------------|--------|
| `full` | 0.968 | 0.915 | 0.914 | 0.851 | **1,595** | $0.004 |
| `rlm_soft` | **0.981** | **0.955** | **0.961** | **0.856** | 5,431 | $0.002 |
| `fold` | 0.930 | 0.939 | 0.841 | 0.831 | 8,391 | $0.003 |
| `reflector` | 0.958 | 0.916 | 0.911 | 0.851 | 1,399 | $0.004 |

#### GPT-4o-mini (100 questions)

| Strategy | Faithful | Recall | Precision | Relevancy | Total (ms) | $/eval |
|----------|----------|--------|-----------|-----------|------------|--------|
| `full` | 0.947 | 0.918 | 0.915 | 0.865 | **1,691** | $0.002 |
| `rlm_soft` | **0.989** | **0.945** | **0.957** | 0.847 | 5,566 | **$0.001** |
| `fold` | 0.946 | 0.928 | 0.847 | 0.850 | 8,961 | $0.001 |
| `reflector` | 0.956 | 0.916 | 0.915 | **0.865** | 1,708 | $0.002 |

### Key Finding 1: Haiku is more faithful than Sonnet

Counter-intuitive result: Claude Haiku (0.985 faithfulness) outperforms Claude Sonnet (0.958) on faithfulness. Haiku sticks closer to the provided context, while Sonnet adds elaboration that, while helpful to the user, can introduce information not strictly present in the retrieved chunks. For applications where grounding is critical (medical, legal, compliance), Haiku may be the better choice despite being the "smaller" model.

### Key Finding 2: Soft compression improves recall AND precision

Across 3 trials, `rlm_soft` (0.814) and `skill_progressive` (0.810) achieve the best recall, both outperforming the `full` baseline (0.776). The soft compressor reorganizes context without stripping it, while skill classification routes to targeted sources. The tradeoff is latency: the Haiku compression call adds ~3.6 seconds, and classification adds ~500ms.

### Key Finding 3: Context packing is the best multi-hop strategy

`context_pack` achieves the highest multi-hop faithfulness (0.960 vs 0.871 for `full`) and the highest overall faithfulness (0.967). The always-on knowledge index gives the LLM background context that helps cross-reference facts across documents. The cost is severe latency: 11s total due to the Haiku sub-LLM call.

### Key Finding 4: Skill classification hurts more than it helps

`skill_progressive` (0.911 faithfulness) underperforms the baseline `full` (0.946). The classification step misroutes ambiguous queries, and it struggles particularly on unanswerable questions (0.728 faithfulness) where it routes to irrelevant sources that produce partially confident responses. It also has the lowest pass^3 reliability (0.618), meaning only a 62% chance of 3/3 consistent good answers.

### Key Finding 5: GPT-4o is 2x faster than Sonnet

GPT-4o at 1.6 seconds total response time vs Sonnet's 3.5 seconds, with comparable accuracy (0.968 vs 0.946 faithfulness). For latency-critical applications (e-commerce, consumer chatbots), this is the difference between "PROD READY" and "TOO SLOW" on the same strategy. GPT-4o-mini's `rlm_soft` achieves the highest faithfulness of any config/model combination (0.989) at just $0.001/eval.

### Key Finding 6: RAG systems are 82% deterministic at temperature=0

Running 3 independent trials on identical questions reveals that 82% of questions produce identical scores every time. But 4% of questions have >0.2 faithfulness spread across trials, where the same question can score 0.8 on one run and 0.5 on another. The non-determinism comes primarily from HNSW approximate nearest neighbors and concentrates on harder question types (unanswerable: 0.086 mean spread vs single-hop: 0.006).

### Performance by Question Type (Sonnet, faithfulness, averaged across 3 trials)

| Strategy | Single-hop | Multi-hop | Temporal | Recall | Adversarial | Unanswerable |
|----------|------------|-----------|----------|--------|-------------|-------------|
| `full` | 0.987 | 0.871 | 0.939 | 0.947 | **0.985** | 0.925 |
| `context_pack` | **0.996** | **0.960** | 0.934 | 0.929 | **0.989** | 0.913 |
| `rlm_soft` | **0.998** | 0.842 | 0.907 | **0.996** | 0.940 | 0.813 |
| `skill_progressive` | 0.962 | 0.867 | 0.873 | 0.888 | 0.936 | 0.728 |

All strategies handle single-hop questions well (> 0.96 faithfulness). The differentiation comes on harder types:
- **Multi-hop**: `context_pack` leads significantly (0.960 vs 0.871 for `full`) because the passive knowledge index helps cross-reference facts across documents
- **Recall**: `rlm_soft` excels (0.996) since compression reorganizes retrieved context to surface relevant facts more effectively
- **Temporal**: All strategies struggle, and conversation history awareness remains the hardest capability
- **Adversarial**: `context_pack` and `full` lead (~0.99) because simpler pipelines are less likely to introduce confusion
- **Unanswerable**: `full` handles these best (0.925), while `skill_progressive` struggles (0.728) due to the classification step sometimes routing to irrelevant sources that produce partially confident responses

---

## Results: Multi-Turn Conversations

The single-shot benchmark tells you how well a strategy handles individual questions. The multi-turn benchmark tells you how well it handles *conversations*, which is what production agents actually do.

### Context Retention

Does the agent remember facts from early turns when asked about them later?

In the Long Horizon scenario (15 turns), we introduce 10 facts in turns 1-10, then test recall of early facts in turns 11-15.

| Strategy | Turns 1-3 | Turns 4-6 | Turns 7-10 | Turns 11-15 | Degradation |
|----------|-----------|-----------|------------|-------------|-------------|
| `full` | 0.97 | 0.93 | ~0.90 | ~0.85 | -0.12 |
| `fold` | TBD | TBD | TBD | TBD | TBD |
| `reflector` | TBD | TBD | TBD | TBD | TBD |
| `agentfold` | TBD | TBD | TBD | TBD | TBD |

Early results show `full` degrading by ~12% from early to late turns. The stateful strategies (fold, reflector, agentfold) are designed to mitigate this: fold by maintaining an evolving summary, reflector by accumulating lessons, agentfold by building hierarchical memory.

### Latency Degradation

Does response time grow as conversations get longer?

| Strategy | Turn 1 (ms) | Turn 5 (ms) | Turn 10 (ms) | Turn 15 (ms) | Growth |
|----------|-------------|-------------|--------------|--------------|--------|
| `full` | 3,600 | 5,600 | ~6,500 | ~7,500 | 2.1x |
| `fold` | TBD | TBD | TBD | TBD | TBD |

With `full`, latency grows ~2x over 15 turns as conversation history accumulates in the LLM context. Strategies that compress or fold context should show flatter growth curves.

### Turn Type Analysis

| Turn Type | What It Tests | `full` Faithfulness |
|-----------|---------------|---------------------|
| simple_recall | Basic fact retrieval | 1.00 |
| follow_up | Reference resolution ("it", "they") | 1.00 |
| backref | Recall of earlier facts | 1.00 |
| inference | Deriving conclusions from facts | 0.90 |
| synthesis | Combining multiple facts into one answer | 0.90 |
| counterfactual | "What if X instead of Y?" | 0.85 |
| correction | Fixing false premises | 0.80 |
| full_recall | Summarize everything discussed | 0.80 |
| full_recall_after_adversarial | Recall after being misled | 0.80 |

The hardest capabilities are counterfactual reasoning, adversarial correction, and full conversation summarization. These are where memory architecture differences matter most.

---

## Production Readiness

### Latency SLA Compliance

What percentage of queries meet each latency target?

| Strategy | < 1.5s | 1.5-3s | 3-5s | > 5s | Total P95 (ms) | Verdict |
|----------|--------|--------|------|------|----------------|---------|
| `full/gpt-4o` | 0% | 60% | 35% | 5% | 2,000 | PROD READY |
| `full/gpt-4o-mini` | 0% | 50% | 40% | 10% | 2,200 | PROD READY |
| `full/sonnet` | 0% | 28% | 65% | 6% | 5,278 | ACCEPTABLE |
| `skill_progressive/sonnet` | 0% | 16% | 69% | 14% | 5,700 | ACCEPTABLE |
| `rlm_soft/sonnet` | 0% | 0% | 1% | 98% | 9,614 | TOO SLOW |
| `context_pack/sonnet` | 0% | 0% | 9% | 90% | 12,888 | TOO SLOW |

### Time to First Token (TTFT)

For streaming UIs, TTFT matters more than total latency because it's when the user sees the first character.

| Strategy | TTFT P50 (ms) | TTFT P95 (ms) |
|----------|--------------|--------------|
| `skill_progressive/sonnet` | 977 | 1,656 |
| `rlm_soft/sonnet` | 986 | 2,018 |
| `full/sonnet` | 996 | 1,711 |
| `context_pack/sonnet` | 6,305 | 7,044 |

`full`, `rlm_soft`, and `skill_progressive` all achieve sub-1s median TTFT because the retrieval completes quickly and the LLM starts streaming immediately. `context_pack` pays 6x more because it waits for the Haiku knowledge-index call before the main LLM can start.

### Retrieval Latency (What We Control)

| Strategy | P50 (ms) | P95 (ms) |
|----------|----------|----------|
| `full` | 514 | 759 |
| `rlm_soft` | 598 | 965 |
| `skill_progressive` | 1,075 | 1,865 |
| `context_pack` | 2,779 | 3,121 |

The `context_pack` retrieval includes building the knowledge index (Neo4j triples + Haiku conversation compression), which adds ~2.2 seconds. The `skill_progressive` retrieval includes a Haiku classification call (~500ms). `rlm_soft` has the same retrieval as `full` since compression happens post-retrieval.

### Cost Analysis

| Strategy/Model | $/eval | $/1000 queries | Monthly (10K queries/day) |
|----------------|--------|----------------|--------------------------|
| `rlm_compress/haiku` | $0.001 | $1.00 | $300 |
| `full/gpt-4o-mini` | $0.002 | $2.00 | $600 |
| `full/haiku` | $0.003 | $3.00 | $900 |
| `skill_progressive/haiku` | $0.002 | $2.00 | $600 |
| `full/sonnet` | $0.006 | $6.00 | $1,800 |
| `context_pack/sonnet` | $0.007 | $7.00 | $2,100 |

The 7x cost difference between cheapest (`rlm_compress/haiku` at $0.001) and most expensive (`context_pack/sonnet` at $0.007) matters at scale. For high-volume applications, strategy choice directly impacts infrastructure costs.

---

## Results: Token & Cost Efficiency

Not all quality is created equal. A strategy that scores 0.96 faithfulness at $0.001/eval is fundamentally different from one that scores 0.96 at $0.007/eval.

### Pareto Frontier

We compute the quality-cost Pareto frontier: strategies where no other strategy is both higher quality AND cheaper. Only Pareto-optimal strategies represent rational choices; everything else is dominated.

| Priority | Pareto-Optimal Strategy | Faith | $/eval | Why Choose |
|----------|------------------------|-------|--------|------------|
| Cheapest | `rlm_compress/haiku` | 0.939 | $0.001 | 7x cheaper than most, acceptable quality |
| Speed | `full/gpt-4o` | ~0.96 | $0.004 | 1.3s total, fast enough for consumer |
| Balanced | `full/sonnet` | 0.958 | $0.006 | Highest quality baseline |
| Best recall | `skill_progressive/sonnet` | 0.920 | $0.004 | Finds 3% more relevant context |

### Token Efficiency

Measured as faithfulness per 1,000 tokens consumed:

- Compression strategies (`rlm_compress`, `rlm_soft`) achieve the highest token efficiency by squeezing more quality per token through removing noise before the main LLM sees it
- `context_pack` has the lowest token efficiency due to always-on system prompt injection, but achieves the highest relevancy
- Model choice matters more than strategy: `haiku` is ~2x more token-efficient than `sonnet` due to lower token consumption with comparable faithfulness

| Config | Quality/1K tokens | Quality/$ | Quality/sec | Tokens/eval | $/eval | Pareto? |
|--------|------------------|-----------|-------------|-------------|--------|---------|
| `rlm_soft` | **0.389** | **277** | 0.128 | 2,413 | $0.003 | Yes |
| `skill_progressive` | 0.270 | 216 | 0.147 | 3,374 | $0.004 | |
| `full` | 0.195 | 147 | **0.264** | 4,842 | $0.006 | Yes |
| `context_pack` | 0.183 | 124 | 0.099 | 5,292 | $0.008 | Yes |

*All values averaged across 3 independent trials on answerable questions.*

`rlm_soft` delivers nearly 2x the quality per dollar compared to `full` (277 vs 147) because compression strips noise tokens before the main LLM, so you pay for less input. But `full` wins on quality per second (0.264 vs 0.128) because it avoids the compression LLM call entirely. Three of four configs land on the Pareto frontier: `rlm_soft` (cheapest), `full` (fastest), `context_pack` (highest quality).

---

## Results: Inter-Rater Reliability

Our dual-judge system (Haiku + GPT-4o-mini) scores each metric independently. This lets us measure judge agreement:

- **Agreement rate** (within ±0.1): Percentage of evaluations where both judges give scores within 0.1 of each other
- **Pearson correlation**: Linear relationship between judge scores
- **Systematic bias**: Whether one judge consistently scores higher

| Metric | Pearson r | Agreement (±0.1) | Systematic Bias | N |
|--------|-----------|-----------------|-----------------|---|
| Faithfulness | 0.451 | 82.3% | +0.008 (negligible) | 440 |
| Context Recall | 0.870 | 72.7% | +0.033 (Haiku slightly higher) | 440 |
| Context Precision | 0.917 | 81.6% | -0.031 (Mini slightly higher) | 440 |
| Answer Relevancy | 0.728 | 66.4% | **-0.101 (Mini scores higher)** | 440 |

The judges agree well on precision (r=0.917) and recall (r=0.870), but diverge on answer relevancy, where GPT-4o-mini consistently scores relevancy ~0.10 higher than Haiku. This systematic bias means our averaged relevancy scores are robust (the bias cancels), but single-judge evaluations would be misleading. Faithfulness shows the lowest correlation (r=0.451) despite high agreement rate (82%), because most scores cluster near 1.0 leaving little variance to correlate on.

---

## Results: pass@k Consistency

Even at temperature=0, RAG systems have sources of non-determinism:
- **HNSW approximate nearest neighbors**: Different traversal paths may return slightly different results
- **Server-side embedding variance**: Batching effects can produce slightly different vectors
- **Neo4j path ordering**: Graph traversal isn't deterministic

We measure this by running k=3 trials per question:
- **pass@k**: Probability of at least 1 success in k trials, answering "will the agent find a good answer eventually?"
- **pass^k**: Probability of ALL k trials succeeding, answering "is the agent reliably consistent?" A 75% per-trial success rate over 3 trials gives (0.75)^3 = 42%.
- **Consistency score** = 1 - std(faithfulness) across trials
- **Unstable questions**: Where trials disagree by > 0.2 in faithfulness

| Config | Consistency | pass@3 | pass^3 | Unstable (>0.2 spread) | Mean Spread |
|--------|------------|--------|--------|----------------------|-------------|
| `full` | **0.978** | 1.000 | **0.804** | 3/100 (3%) | 0.022 |
| `context_pack` | 0.959 | 1.000 | **0.839** | 6/100 (6%) | 0.042 |
| `rlm_soft` | 0.956 | 0.998 | 0.679 | 7/99 (7%) | 0.044 |
| `skill_progressive` | 0.975 | 0.997 | 0.618 | 3/99 (3%) | 0.025 |

**pass@3** (will it work eventually?) is near-perfect for all configs. If you retry, you'll almost certainly get a good answer. But **pass^3** (will it work every time?) reveals real differences:

- `context_pack` is the most reliable at 0.839, with an 84% chance of 3/3 good answers
- `skill_progressive` drops to 0.618, giving only a 62% chance of consistent success
- `full` achieves 0.804 with the lowest mean spread (0.022), making it the most deterministic pipeline

**What causes instability?** Question difficulty predicts variance:

| Question Type | Mean Spread | Unstable (>0.2) |
|--------------|-------------|-----------------|
| Single-hop | 0.006 | 0% |
| Adversarial | 0.016 | 3% |
| Recall | 0.023 | 3% |
| Multi-hop | 0.040 | 5% |
| Temporal | 0.048 | 7% |
| Unanswerable | 0.086 | 15% |

Simple factoid retrieval is rock-solid. Non-determinism concentrates in harder queries where HNSW may return slightly different neighbors, cascading into different context and different answers. Notably, faithfulness varies more than recall (0.029 vs 0.022 mean spread for `full`), meaning the non-determinism comes more from the LLM's interpretation than from retrieval itself.

We also flag **eval saturation**, where metrics have scores near-ceiling (mean >= 0.95 with 50%+ perfect scores). `faithfulness` on `context_pack` (0.967) is approaching saturation, suggesting we may need harder faithfulness challenges to differentiate top strategies.

---

## Results: Component Ablation

The ablation isolates each retrieval component's marginal contribution. Working upward from a single dense vector search:

*Full component ablation (`bench.ablation`) pending. Partial view from current configs:*

| Component Stack | Faithfulness | Recall | Precision | Ret P50 (ms) | $/eval |
|-----------------|-------------|--------|-----------|-------------|--------|
| Hybrid + Neo4j + PG (`full`) | 0.948 | 0.713 | 0.803 | 514 | $0.006 |
| + Context Pack | 0.959 (+0.011) | 0.713 | 0.801 | 2,779 | $0.008 |
| + RLM Soft Compression | 0.937 (-0.011) | 0.749 (+0.036) | 0.844 (+0.041) | 598 | $0.003 |
| + Skill Classification | 0.887 (-0.061) | 0.736 (+0.023) | 0.782 (-0.021) | 1,075 | $0.004 |

Context packing adds 0.011 faithfulness but 5x retrieval latency because the Haiku sub-call to build the context pack is the bottleneck. Compression improves recall (+0.036) and precision (+0.041) by stripping noise, at the cost of faithfulness (-0.011) and 2x total latency. Skill classification hurts faithfulness (-0.061) due to misrouting on ambiguous queries.

---

## Results: Scale Degradation

How does retrieval quality change as the knowledge base grows from 100 to 1,000,000 documents?

We populated a Qdrant collection with 1,000,029 documents (29 real knowledge base entries buried among 1 million synthetic distractors) and ran the same 100 evaluation questions against it.

| Config | Faithfulness | Context Recall | Context Precision | Answer Relevancy | Ret P50 (ms) | Ret P95 (ms) | Total P50 (ms) |
|---|---|---|---|---|---|---|---|
| full | 0.958 ± 0.024 | 0.784 ± 0.056 | 0.869 ± 0.041 | 0.858 ± 0.043 | 515 | 704 | 3,386 |
| rlm_soft | 0.929 ± 0.030 | 0.818 ± 0.061 | 0.878 ± 0.049 | 0.845 ± 0.043 | 623 | 854 | 7,129 |
| skill_progressive | 0.862 ± 0.043 | 0.789 ± 0.060 | 0.806 ± 0.043 | 0.848 ± 0.046 | 1,105 | 1,746 | 4,242 |

**Key findings at 1M scale:**

- **Retrieval holds up remarkably well.** The `full` hybrid pipeline maintains 0.958 faithfulness and sub-520ms P50 retrieval even with a million documents. Qdrant's HNSW index with INT8 quantization handles the scale without meaningful quality loss.
- **Compression costs more than it saves at scale.** `rlm_soft` achieves marginally better recall (0.818 vs 0.784) but doubles total latency to 7.1 seconds. The LLM compression pass becomes the bottleneck, not retrieval.
- **Classification overhead compounds.** `skill_progressive` drops to 0.862 faithfulness with 2x retrieval latency. The classification step adds ~600ms per query and occasionally misroutes, hurting both speed and quality.
- **The simple approach wins.** At production scale, the straightforward hybrid retrieval pipeline (dense + sparse + graph + history) outperforms strategies that add LLM calls into the retrieval path.

---

## Results: Hallucination Resistance

A reliable agent needs to know what it doesn't know. We added 10 unanswerable questions: queries about information genuinely absent from the knowledge base (revenue figures, employee counts, pricing, third-party integrations). The correct behavior is to abstain rather than hallucinate a confident answer.

We score **abstention**: 1.0 = correctly says "I don't have that information", 0.0 = confidently fabricates an answer.

| Config | Abstention Score | Faithfulness | N (across 3 trials) |
|--------|-----------------|-------------|---------------------|
| `full` | **1.000** | 0.925 | 30 |
| `context_pack` | **1.000** | 0.913 | 27 |
| `rlm_soft` | **1.000** | 0.813 | 30 |
| `skill_progressive` | **1.000** | 0.728 | 30 |

All four strategies scored perfect abstention across 3 independent trials. Sonnet consistently says "I don't have that information" rather than hallucinating. But faithfulness on unanswerable questions varies significantly: `full` scores 0.925 while `skill_progressive` drops to 0.728. The classification step in `skill_progressive` sometimes routes to irrelevant sources that introduce partial hallucinations, even though the final answer still technically abstains.

This is a strong result but also suggests our unanswerable questions may be too easy to distinguish (retrieval returns obviously irrelevant content). Harder unanswerable questions, where retrieved context is *related but insufficient*, would be a more discriminating test.

---

## The Recommended Architectures

Based on our results, here are the recommended configurations by use case:

### For E-Commerce / Consumer Chatbots

**`full` + GPT-4o** or **`full` + GPT-4o-mini**

| Metric | GPT-4o | GPT-4o-mini |
|--------|--------|-------------|
| Total latency | ~1.3s | ~1.4s |
| Faithfulness | ~0.96 | ~0.95 |
| Cost/eval | $0.004 | $0.002 |

Why: Speed is everything. No compression needed because raw retrieval is fast enough, and every sub-LLM call adds 1.5-3 seconds that defeats the purpose. GPT-4o-mini is the best bang for buck if you can tolerate slightly lower quality.

### For Customer Support / Help Desk

**`full` + Sonnet** (simple queries) or **`skill_progressive` + Sonnet** (complex queries)

| Metric | full | skill_progressive |
|--------|------|-------------------|
| Total latency | 3.2s | 3.8s |
| Recall | 0.913 | 0.939 |
| Best at | Direct answers | Multi-hop reasoning |

Why: Sonnet's reasoning quality matters for nuanced customer questions. `skill_progressive` adds ~600ms for a classification step but finds 3% more relevant context, which is worth it when customers ask complex multi-part questions.

### For IT Support / Internal Tools

**`context_pack` + Haiku**

| Metric | Value |
|--------|-------|
| Total latency | ~5s |
| Faithfulness | 0.969 |
| Relevancy | 0.867 (highest) |
| Cost/eval | $0.003 |

Why: Internal users tolerate longer response times (3-5s) in exchange for more complete answers. The passive knowledge index means the agent always has background context about the user's environment. Haiku is 2x cheaper than Sonnet with higher faithfulness.

### For High Volume / Cost Sensitive

**`rlm_compress` + Haiku** or **`full` + GPT-4o-mini**

| Metric | rlm_compress/haiku | full/gpt-4o-mini |
|--------|-------------------|------------------|
| Cost/eval | $0.001 | $0.002 |
| Faithfulness | 0.939 | ~0.95 |
| Latency | 4.4s | 1.4s |
| Best for | Quality at low cost | Speed at low cost |

Why: At 10K+ queries/day, the difference between $0.001 and $0.007 is $1,800/month. `rlm_compress/haiku` is the cheapest option with acceptable quality. If speed matters more than cost, GPT-4o-mini is 3x faster at 2x the price.

### For Maximum Accuracy

**`skill_progressive_pack` + Sonnet**

| Metric | Value |
|--------|-------|
| Recall | 0.949 (highest) |
| Relevancy | 0.877 (tied highest) |
| Latency | 5.5s |
| Cost/eval | $0.005 |

Why: Progressive disclosure finds the right context, and the passive pack ensures background knowledge is always present. The combination achieves the highest recall of any configuration. Not fast enough for consumer-facing use, but ideal for analyst tools, research assistants, or compliance systems.

### For Multi-Turn Conversations

**`fold`** or **`reflector`** (preliminary recommendation)

Why: Stateful strategies designed for multi-turn use. `fold` maintains an incremental summary that prevents context rot over long conversations. `reflector` accumulates lessons from each turn, improving later answers. Both are new and results are preliminary. We'll update these recommendations as multi-turn benchmarks complete.

---

## The Decision Matrix

| Your Priority | Strategy | Model | $/eval | Latency | Quality Score |
|---------------|----------|-------|--------|---------|---------------|
| **Speed** | `full` | GPT-4o | $0.004 | 1.3s | 0.94 |
| **Cheapest** | `rlm_compress` | Haiku | $0.001 | 4.4s | 0.93 |
| **Balanced** | `full` | Sonnet | $0.006 | 3.2s | 0.96 |
| **Best recall** | `skill_progressive_pack` | Sonnet | $0.005 | 5.5s | 0.95 |
| **Best relevancy** | `context_pack` | Sonnet | $0.007 | 6.1s | 0.96 |
| **Multi-turn** | `fold` | Sonnet | ~$0.004 | ~6s | TBD |
| **Self-improving** | `reflector` | Sonnet | ~$0.006 | ~5.5s | TBD |
| **Budget speed** | `full` | GPT-4o-mini | $0.002 | 1.4s | 0.93 |

Quality Score = average of faithfulness and recall.

---

## Try It Yourself

The benchmark suite is open source and designed to run against your own Qdrant data. Every result in this article can be reproduced:

```bash
# Install dependencies
uv sync

# Run single-shot benchmark (all strategies, Sonnet)
uv run python -m bench.runner --configs full,context_pack,rlm_compress,skill_progressive

# Compare models
uv run python -m bench.runner --configs full --models sonnet,haiku,gpt-4o,gpt-4o-mini

# Run with pass@k consistency (3 trials per question)
uv run python -m bench.runner --configs full,rlm_soft --trials 3

# Run multi-turn conversation scenarios
uv run python -m bench.env_runner --configs full,fold,reflector --scenarios long_horizon

# Component ablation study
uv run python -m bench.ablation --model sonnet --limit 50

# Statistical analysis on existing results
uv run python -m bench.stats results/raw_YYYYMMDD_HHMMSS.json

# Scale benchmark (requires pre-populated scale collection)
uv run python -m bench.scale_data --scale 10k
uv run python -m bench.scale_runner --configs full,rlm_soft --model sonnet
```

### Adapting to Your Data

1. **Replace the eval dataset**: Edit `bench/eval_dataset.json` with questions and ground truths from your domain
2. **Add conversation scenarios**: Edit `bench/conversations.py` with multi-turn flows from your use case
3. **Point to your Qdrant collection**: Update `.env` with your Qdrant URL and collection name
4. **Run and compare**: The benchmark produces markdown reports with production readiness verdicts

The strategies are also available as a runtime API:

```python
# POST /chat
{
    "message": "What is your refund policy?",
    "strategy": "skill_progressive"  # or "full", "context_pack", etc.
}

# GET /strategies - list all available strategies with benchmark scores
```

---

## What We Learned

### Layered Context Beats More Context

The central insight from [skill.md meets REPL](https://qdrant.tech/articles/skill-md-meets-repl/) holds up at scale: agents need the *right* context at the *right layer*, not just more context. Our results confirm this across every strategy:

- `skill_progressive` beats `full` on recall not by retrieving more, but by classifying first (the skill.md "known unknowns" layer) then looking in the right place (the REPL "unknown unknowns" layer)
- `context_pack` beats `full` on relevancy by front-loading knowledge into the system prompt, the same principle as injecting a skill file before the agent starts working
- `rlm_compress` trades recall for precision by adding a compression layer between retrieval and generation

The strategies that add undifferentiated context (`context_pack_rlm` at 12 seconds, `skill_progressive_pack` at 5.5 seconds) pay the latency cost without proportional quality gains. The strategies that add *targeted* context at the right layer (`skill_progressive` at 3.8 seconds, `context_pack` at 6.1 seconds) deliver measurable improvements.

### The Bitter Lesson Applies to Memory Too

Simple strategies with more compute (bigger model, more retrieval) often beat clever strategies with less compute. `full/sonnet` outperforms `rlm_compress/sonnet` despite the latter having a sophisticated compression pipeline. The compression adds complexity and latency while removing information the LLM needs.

However, there are important exceptions:
- `skill_progressive` beats `full` on recall by being smarter about *where* to look, not by adding compute
- `context_pack` beats `full` on relevancy by giving the LLM ambient knowledge
- Compression wins on cost, with `rlm_compress/haiku` coming in 6x cheaper than `full/sonnet`

### Faithfulness and Elaboration Are in Tension

Sonnet elaborates more than Haiku, which users prefer in conversation but which reduces faithfulness scores. This creates a real tradeoff: do you want an agent that sticks strictly to the provided context (Haiku, 0.985 faithfulness), or one that provides richer, more helpful responses that occasionally go slightly beyond the evidence (Sonnet, 0.958)?

For most customer support applications, Sonnet's slight faithfulness penalty is worth it. For medical, legal, or compliance applications, Haiku's strict grounding is safer.

### Latency Is the Strongest Constraint

In our testing, latency is the factor most likely to eliminate a strategy from consideration. Context packing improves quality by every metric, but its 6-second response time makes it unsuitable for consumer-facing applications. The fastest path to production readiness is often choosing a faster model (GPT-4o) rather than a simpler strategy.

### There Is No Best Strategy

This is the central finding. The right memory architecture depends on:
- **Latency budget**: Consumer (< 2s) vs internal (< 5s) vs batch (doesn't matter)
- **Cost sensitivity**: $0.001/eval vs $0.007/eval is 7x at scale
- **Quality priority**: Recall (find everything) vs precision (reduce noise) vs faithfulness (stay grounded)
- **Conversation style**: Single-shot vs multi-turn vs long-horizon

The buffet approach of offering multiple strategies and letting users choose is not a cop-out. It's the architecturally correct solution.

---

## Future Work

- **Complete multi-turn benchmarks**: Fold, AgentFold, and Reflector results on all conversation scenarios
- **Streaming latency**: Measure time-to-first-token instead of total response time, since perceived speed differs from actual speed
- **RL on strategy selection**: Train a model to automatically pick the best strategy based on query characteristics
- **Scale degradation curves**: Run all strategies at 10K, 100K, and 1M documents to measure quality and latency degradation
- **Fine-tuned compressor**: Train a small model specifically for context compression instead of using general-purpose Haiku
- **Cross-session memory**: Test strategies where knowledge persists across separate conversations, not just within one

---

*Built with [Qdrant](https://qdrant.tech) for vector search, Neo4j for knowledge graphs, and PostgreSQL for conversation history. Benchmarked across Claude Sonnet, Claude Haiku, GPT-4o, and GPT-4o-mini. Statistical analysis includes bootstrap CIs, paired significance tests, inter-rater reliability, and pass@k consistency.*
