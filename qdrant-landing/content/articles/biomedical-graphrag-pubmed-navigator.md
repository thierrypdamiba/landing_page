---
title: "Building a PubMed Navigator: Engineering a Biomedical GraphRAG System with Qdrant"
short_description: "How we combined Qdrant hybrid search, Neo4j knowledge graphs, and agentic tool routing to build a medical literature copilot — with production query results."
description: "A developer walkthrough of building a biomedical GraphRAG system that combines Qdrant vector search (hybrid retrieval, constraint-based recommendations, scalar quantization) with Neo4j knowledge graphs and LLM-driven tool selection to navigate PubMed literature."
preview_dir: /articles_data/biomedical-graphrag-pubmed-navigator/preview
social_preview_image: /articles_data/biomedical-graphrag-pubmed-navigator/preview/social_preview.png
weight: -160
author: Thierry Damiba
author_link: https://www.linkedin.com/in/thierry-damiba
date: 2026-01-28T00:00:00.000Z
category: rag-and-genai
---

Healthcare AI adoption is accelerating. According to [Menlo Ventures](https://menlovc.com/perspective/2025-the-state-of-ai-in-healthcare/), 22% of healthcare organizations in the US have deployed domain-specific AI tools — from ambient scribes that handle patient documentation to research copilots that help navigate biomedical literature. What these tools share is a dependency on **context engineering**: getting the right information to the right model at the right time.

My colleague Jenny recently put out a [video](/TODO) walking through the motivation behind our PubMed Navigator and the Qdrant-related architectural choices we made. This article is the engineering companion — how we actually wired it together, what the code looks like, and what happens when real queries hit the system.

## The Problem: PubMed Is Huge and Keyword Search Isn't Enough

[PubMed](https://pubmed.ncbi.nlm.nih.gov/about/) is a semi-structured knowledge base with over 39 million citations and abstracts of biomedical literature. Scientists search it for **evidence** (papers), **patterns** (what genes, drugs, and diseases co-occur), and **confidence** (is a finding well-studied or based on a single paper?).

Traditionally, PubMed's APIs offer keyword/full-text search and are rate-limited. That works for simple lookups, but consider these four queries:

1. *"What is the relationship between APOE variants and Alzheimer's disease progression?"*
2. *"What is the relationship between APOE variants and Alzheimer's disease, excluding studies focused on cardiovascular outcomes?"*
3. *"Which researchers have co-published on APOE and neurodegeneration?"*
4. *"What genes are frequently co-mentioned with BRCA1 in breast cancer research?"*

Full-text matching on "APOE" + "Alzheimer's" can approximate query 1, but fails on the rest. Query 2 requires navigating **semantic dissimilarity** — finding papers *toward* a concept and *away from* another. Query 3 requires **relational intelligence** — traversing author-paper-author connections across the literature. Query 4 requires **co-occurrence analysis** — finding which genes appear alongside BRCA1 in papers about breast cancer.

We ran all four queries against our production system. The results demonstrate exactly why you need two complementary retrieval systems.

## Why Vector Search

Vector search encodes text into high-dimensional embedding space, where **semantic similarity** maps to geometric proximity. Papers about "APOE epsilon-4 allele and amyloid beta accumulation" and "apolipoprotein E polymorphisms in cognitive decline" land near each other in this space, even though they share few keywords.

More importantly, vector space supports **constraint-based retrieval**. Qdrant's [recommendation API](/documentation/concepts/explore/#recommendation-api) finds points *similar to* positive examples and *dissimilar to* negative examples. This is how query 2 works: embed "APOE Alzheimer's disease progression" as a positive and "cardiovascular outcomes" as a negative, then find papers in the region satisfying both constraints. This is fundamentally unsolvable with keyword search.

Here's the proof. When we ran query 1 (no constraints), the system used `retrieve_papers_hybrid` and returned:

| Paper | Score |
|---|---|
| Association of Rare APOE Missense Variants V236E and R251G With Risk of Alzheimer Disease (PMID: 35639372) | 0.603 |
| Multi-functional role of apolipoprotein E in neurodegenerative diseases (PMID: 39944166) | 0.600 |
| APOE Christchurch enhances a disease-associated microglial response to plaque... (PMID: 39844286) | 0.597 |

When we ran query 2 (excluding cardiovascular studies), the system switched tools to `recommend_papers_based_on_constraints` with `positive_examples: ["relationship between APOE variants and Alzheimer disease"]` and `negative_examples: ["studies focused on cardiovascular outcomes"]`. The results shifted:

| Paper | Score |
|---|---|
| Multi-functional role of apolipoprotein E in neurodegenerative diseases (PMID: 39944166) | 0.515 |
| APOE genotype effects on Alzheimer's disease onset and epidemiology (PMID: 15181244) | 0.493 |
| APOE Christchurch enhances a disease-associated microglial response... (PMID: 39844286) | 0.475 |

The rare variants paper (PMID: 35639372) — which discusses APOE's role in lipid transport alongside AD risk — dropped out entirely. The APOE genotype epidemiology paper (PMID: 15181244), which is purely about AD onset with no cardiovascular dimension, took its place. The constraint reshaped the result set. Different tool, different papers, different scores.

## Why Knowledge Graphs

Graphs encode **relationships** between entities — authors, papers, genes, MeSH terms — as nodes and edges. For queries 3 and 4, you need to traverse paths like:

- **Author → Paper (MeSH term) ← Author** (co-authorship filtered by topic)
- **Gene → Paper (MeSH term) ← Gene** (gene co-occurrence in literature)
- **Paper → MeSH term ← Paper** (topically related papers)

These are multi-hop relational queries. Vector embeddings compress a paper's content into a single point — they don't preserve the structured relationships between entities *across* papers. Graphs do.

When query 3 hit the system, it first ran `retrieve_papers_hybrid` to find relevant APOE/neurodegeneration papers. From the results, the agent extracted author "Henrik Zetterberg" and topics `["Apolipoproteins E", "neurodegeneration"]` from the Qdrant payload, then called `get_collaborators_with_topics`. The graph traversal returned:

| Collaborator | Shared Papers | Topics |
|---|---|---|
| Kaj Blennow | 2 | Apolipoproteins E |
| Charlotte E Teunissen | 2 | Apolipoproteins E |
| Hilkka Soininen | 2 | Apolipoproteins E |
| Tormod Fladby | 2 | Apolipoproteins E |
| Anna Zettergren | 2 | Apolipoproteins E |

No embedding similarity search can produce this. These are multi-hop traversals: Author → WROTE → Paper → HAS_MESH_TERM → "Apolipoproteins E" ← HAS_MESH_TERM ← Paper ← WROTE ← co-Author.

Query 4 demonstrates a different graph capability. The agent ran `retrieve_papers_hybrid` to find BRCA1/breast cancer papers, then extracted gene "BRCA1" and MeSH term "Breast Neoplasms" from the payload and called `get_genes_in_same_papers`. The graph traversal returned genes that co-occur with BRCA1 in breast cancer literature:

| Co-mentioned Gene | Shared Papers | Example PMIDs |
|---|---|---|
| EPRS1 | 4 | 26352356, 27702988, 32586150, 35247837 |
| AGO1 | 4 | 26352356, 27702988, 28986106, 35247837 |
| ATF2 | 4 | 26352356, 27702988, 30753908, 35247837 |
| E2F4 | 4 | 26352356, 27702988, 30753908, 35247837 |
| NBR1 | 4 | 21590391, 26352356, 27702988, 35247837 |
| EXO1 | 4 | 26352356, 27702988, 35247837, 38255897 |

This traverses Gene → MENTIONED_IN → Paper ← MENTIONED_IN ← Gene, filtered by papers tagged with "Breast Neoplasms." The result reveals potential biological associations based on co-mention frequency — which genes show up alongside BRCA1 across the breast cancer literature. No single paper's embedding encodes this; it's a structural pattern across documents.

The core insight: **vector search explores semantic space; graphs verify structural connections.** A biomedical copilot needs both.

## Data Pipeline: Getting 30K Papers Into Shape

Before retrieval can happen, data needs to live in two places: Qdrant (for semantic search) and Neo4j (for relational queries). These aren't redundant — they serve fundamentally different query shapes.

### What goes into Qdrant

Each PubMed paper becomes a point with **three vector representations**:

```python
point = models.PointStruct(
    id=int(pmid),
    vector={
        "Dense": retriever_vector,    # 1536-dim OpenAI embedding
        "Reranker": reranker_vector,  # 3072-dim OpenAI embedding
        "Lexical": sparse_vector,     # BM25 sparse vector
    },
    payload={
        "paper": paper_model.model_dump(),
        "genes": [g.model_dump() for g in pmid_to_genes.get(pmid, [])],
    },
)
```

The Dense and Reranker vectors come from the **same model** (OpenAI's `text-embedding-3-large`) at different dimension truncations. This works because of **Matryoshka Representation Learning** (MRL) — a training technique where the first N dimensions of a larger embedding are themselves a valid embedding. The first 1536 dimensions of a 3072-dim vector encode a coarser but complete semantic representation. The practical implication: **one embedding API call per abstract**, then slice.

```python
openai_vector = await self._get_openai_vectors(
    abstract, dimensions=3072  # full precision
)
retriever_vector = openai_vector[:1536]  # truncated for fast retrieval
reranker_vector = openai_vector          # full precision for reranking
```

When using **[Qdrant Cloud Inference](/documentation/cloud/cloud-inference/)**, this changes — we pass `models.Document` objects instead, and Qdrant handles embedding generation server-side, eliminating the round-trip of shipping vectors from OpenAI to our backend to Qdrant.

### BM25 average length estimation

BM25's scoring formula normalizes term frequency by average document length. Rather than hardcoding a guess, we estimate it from the first N abstracts in the dataset:

```python
for paper in papers:
    if paper.get("abstract"):
        total_words += len(paper["abstract"].split())
        sampled_count += 1
        if sampled_count >= self.estimate_bm25_avg_len_on_x_docs:
            break
avg_abstracts_len = total_words // sampled_count if sampled_count > 0 else 256
```

If this value is wildly off, BM25 ranking distorts. For PubMed abstracts, we measured roughly 200-300 words.

### What goes into Neo4j

The knowledge graph captures relationships that vectors can't encode. From the same PubMed dataset, we extract nodes (Author, Paper, Gene, MeSH term) and edges (WROTE, HAS_MESH_TERM, MENTIONS). This is what enables traversals like the Zetterberg collaborator query and the BRCA1 gene co-occurrence query — connecting entities through shared papers filtered by MeSH topics.

## The Hybrid Retrieval Pipeline

**[Hybrid search](/documentation/concepts/hybrid-queries/)** combines dense (semantic) and sparse (lexical) retrieval into a single query. Dense embeddings catch semantic similarity — synonyms, paraphrases, related mechanisms. BM25 catches exact lexical matches — gene names like APOE and BRCA1, drug identifiers, specific acronyms like "AD" for Alzheimer's disease. Medical literature is full of precise terminology where you want *both*.

Here's the implementation:

```python
search_result = await self.qdrant_client.client.query_points(
    collection_name=self.collection_name,
    prefetch=[
        models.Prefetch(
            query=retriever_vector,        # 1536-dim dense
            using="Dense",
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    oversampling=3.0,
                    rescore=True,
                )
            ),
            limit=top_k,
        ),
        models.Prefetch(
            query=sparse_vector,           # BM25 sparse
            using="Lexical",
            limit=top_k,
        ),
    ],
    query=reranker_vector,                 # 3072-dim reranker
    using="Reranker",
    limit=top_k,
)
```

Two stages:

1. **Prefetch**: Two parallel retrievals — dense (quantized INT8, with 3x oversampling and rescoring against original vectors) and lexical (BM25 with IDF modifier). Each returns `top_k` candidates.
2. **Fusion via reranking**: The union of both candidate sets is re-scored against the 3072-dim Reranker vector. This isn't RRF or a weighted sum — it's a full cosine similarity re-rank using higher-dimensional embeddings that encode finer semantic distinctions.

Design details:

- **Dense** vectors use **[scalar quantization](/documentation/guides/quantization/#scalar-quantization)** (INT8) to reduce RAM. With oversampling=3.0, we retrieve 3x more candidates from the quantized index, then rescore against originals — recovering nearly all accuracy lost from compression.
- **Reranker** vectors are stored **on disk** with `hnsw_config=HnswConfigDiff(m=0)` — no HNSW graph index built. They're only used for rescoring, never primary retrieval. This saves significant RAM.
- **Lexical** vectors use Qdrant's built-in BM25 with IDF modifier, generated server-side.

## Constraint-Based Recommendation

The second Qdrant tool, `recommend_papers_based_on_constraints`, handles queries with positive *and* negative constraints:

```python
query=models.RecommendQuery(
    recommend=models.RecommendInput(
        positive=positive_vectors,
        negative=negative_vectors,
        strategy=models.RecommendStrategy.AVERAGE_VECTOR,
    )
)
```

`AVERAGE_VECTOR` computes a centroid of positive examples and an anti-centroid of negatives, then finds points maximizing similarity to the positive centroid while minimizing similarity to the negative one.

We saw this in action with query 2. The agent decomposed "APOE variants and Alzheimer disease, excluding cardiovascular outcomes" into:
- **positive**: `["relationship between APOE variants and Alzheimer disease"]`
- **negative**: `["studies focused on cardiovascular outcomes"]`

The result: papers about APOE's role in neurodegeneration surfaced, while papers discussing APOE's overlapping role in lipid transport and cardiovascular risk were pushed away — even though many APOE papers legitimately cover both domains.

## The Tool Layer: How the Agent Picks Its Path

We defined **five tools** as OpenAI function-calling schemas — two for Qdrant, three for Neo4j. The LLM selects which tool to invoke and fills in arguments. The backend dispatches.

### Qdrant tools

| Tool | When the agent picks it |
|---|---|
| `retrieve_papers_hybrid(query)` | Pure similarity — "find papers about X" |
| `recommend_papers_based_on_constraints(positive, negative)` | Constraints — "like X, not like Y" |

### Neo4j tools

| Tool | When the agent picks it |
|---|---|
| `get_collaborators_with_topics(author, topics)` | Author collaboration filtered by MeSH topics |
| `get_related_papers_by_mesh(pmid)` | Papers sharing MeSH terms with a retrieved paper |
| `get_genes_in_same_papers(target_gene, mesh_filter)` | Gene co-occurrence filtered by topic |

### The agentic loop in practice

Here's what actually happened with our four production queries:

**Query 1** — *"What is the relationship between APOE variants and Alzheimer's disease progression?"*

```
Step 1: Agent selects → retrieve_papers_hybrid
        Arguments: {query: "relationship between APOE variants and
                    Alzheimer disease progression"}
Step 2: Qdrant returns 3 papers with metadata
Step 3: Agent extracts author "Frank Jessen" and topics
        ["Alzheimer Disease", "Apolipoproteins E"]
        Agent selects → get_collaborators_with_topics
Step 4: Neo4j returns 10 collaborators
        (Hilkka Soininen, Philippe Amouyel, Ole A Andreassen...)
Step 5: Agent selects → get_related_papers_by_mesh(pmid="35639372")
Step 6: Neo4j returns 10 related papers by shared MeSH terms
Step 7: Agent synthesizes all context into grounded answer
```

**Query 2** — *"...excluding studies focused on cardiovascular outcomes"*

```
Step 1: Agent selects → recommend_papers_based_on_constraints  ← DIFFERENT TOOL
        Arguments: {
          positive: ["relationship between APOE variants and Alzheimer disease"],
          negative: ["studies focused on cardiovascular outcomes"]
        }
Step 2: Qdrant returns 3 papers — different set, different scores
Step 3: Agent extracts author "Vivek Swarup"
        Agent selects → get_collaborators_with_topics
Step 4: Neo4j returns collaborators
        (Samuel Morabito, Emily Miyoshi...)  ← DIFFERENT NETWORK
Step 5: Agent synthesizes
```

**Query 3** — *"Which researchers have co-published on APOE and neurodegeneration?"*

```
Step 1: Agent selects → retrieve_papers_hybrid
        Arguments: {query: "co-publishing researchers on APOE and
                    neurodegeneration"}
Step 2: Qdrant returns 3 papers
        (CSF proteomics study, ApoE review, APOE Christchurch)
Step 3: Agent extracts author "Henrik Zetterberg" and topics
        ["Apolipoproteins E", "neurodegeneration"]
        Agent selects → get_collaborators_with_topics
Step 4: Neo4j returns collaborators
        (Kaj Blennow, Charlotte E Teunissen, Hilkka Soininen...)
Step 5: Agent synthesizes
```

**Query 4** — *"What genes are frequently co-mentioned with BRCA1 in breast cancer research?"*

```
Step 1: Agent selects → retrieve_papers_hybrid
        Arguments: {query: "genes frequently co-mentioned with BRCA1
                    in breast cancer research"}
Step 2: Qdrant returns 3 papers
        (BRCA1/2 review, BRCA1 cell line mutations, BRCA1/2 CNAs)
Step 3: Agent extracts gene "BRCA1" and MeSH term "Breast Neoplasms"
        Agent selects → get_genes_in_same_papers  ← GENE CO-OCCURRENCE TOOL
Step 4: Neo4j returns 10 co-mentioned genes
        (EPRS1, AGO1, ATF2, E2F4, NBR1, EXO1...)
Step 5: Agent synthesizes
```

Four things to notice:

1. **Tool routing works.** Query 2 triggered `recommend_papers_based_on_constraints` while queries 1, 3, and 4 triggered `retrieve_papers_hybrid`. The agent correctly identified the negation constraint.

2. **Neo4j arguments are pre-filled from Qdrant results.** The agent didn't hallucinate author names or gene identifiers. It extracted "Frank Jessen," "Vivek Swarup," "Henrik Zetterberg," and "BRCA1" from the paper metadata returned by Qdrant, then passed them to the appropriate graph tool.

3. **Different queries activate different Neo4j tools.** Query 1 triggered both `get_collaborators_with_topics` and `get_related_papers_by_mesh`. Query 4 triggered `get_genes_in_same_papers`. The agent matched each query's intent to the right graph traversal.

4. **Graph reveals cross-document structure.** Hilkka Soininen appeared as a collaborator in both query 1 (via Frank Jessen) and query 3 (via Henrik Zetterberg) — revealing her as a central node in the APOE/Alzheimer's research network. The BRCA1 gene co-occurrence results (EPRS1, AGO1, EXO1) surface potential biological associations that no single paper's embedding can encode.

### Why five narrow tools, not one broad one?

We tried a single "search everything" tool with many parameters. It performed worse. The LLM makes better selections when each tool has a **clear, narrow purpose** with an unambiguous description. Five focused tools with obvious use cases outperform one Swiss-army-knife tool where the model has to figure out which parameters to fill and which to leave empty.

## Conditional Updates for Incremental Ingestion

PubMed is a living dataset. Re-ingesting 30K papers every time you add new ones is wasteful. We use Qdrant's **[conditional updates](/documentation/concepts/points/#conditional-updates)** to upsert only papers that don't already exist:

```python
models.UpsertOperation(
    upsert=PointsList(
        points=[point],
        update_filter=models.Filter(
            must_not=[
                models.HasIdCondition(has_id=[point_id]),
            ]
        )
    )
)
```

This runs via `batch_update_points`, processing conditional upserts atomically per batch. Combined with exponential backoff retry logic (5 attempts, 2s base delay), ingestion handles transient failures gracefully — important when embedding thousands of abstracts through an external API.

## Planning Filters Early

Qdrant's vector index and filtering work together through **[filterable HNSW](/articles/filtrable-hnsw/)** and ACORN algorithms. If you know you'll need filtering — by publication year, MeSH tags, study type — configure those filter indexes **before** building the vector index. Retrofitting works, but you get better performance when the HNSW graph is constructed with filter awareness from the start.

For a biomedical system, obvious filter candidates:
- Publication date range
- MeSH term categories
- Journal name
- Study type (clinical trial, review, meta-analysis)

We haven't fully exploited this yet, but it's on the roadmap.

## Frontend: Transparency as a Feature

Medical researchers need to **trust** the system, which means seeing *why* it returned what it returned. The assistant UI shows:

- **Which tools were executed** and how many matches each returned
- **Vector search results** with matched papers and relevance scores
- **Graph insights** as a separate section (collaborators, co-mentioned genes)
- **A synthesis** combining both sources with inline citations

If a researcher can't trace an answer back to its sources, the tool is useless regardless of how good the retrieval is.

## What We'd Do Differently

**Gene embeddings.** Right now, genes are payload metadata on papers. We don't embed gene descriptions as their own vectors — so you can't query "find papers about genes functionally similar to APOE." That's a separate vector space worth building.

**Evaluation.** We don't yet have a systematic way to measure retrieval quality against a medical ground truth. Every architecture decision here is informed by intuition and qualitative testing. This is the most important gap.

**Recommendation beyond negation.** The constraint tool currently handles explicit exclusion queries. But it could also power "more like this" from a paper the user selects — using the paper's existing vector as a positive example rather than embedding a new text description.

## Try It

The demo is live at [biomedical-graphrag.vercel.app/assistant](https://biomedical-graphrag.vercel.app/assistant). The source code is available on GitHub: [backend](https://github.com/thierrypdamiba/biomedical-graphrag) and [frontend](https://github.com/thierrypdamiba/biomedical-graphrag-frontend). The project started from [Benito Martin's open-source GraphRAG contribution](https://github.com/benitomartin/biomedical-graphrag), which provided the foundation we built on.

If you're building in medtech and have opinions on retrieval quality, evaluation methodology, or graph schema design — reach out in our [Discord community](https://qdrant.to/discord) or on LinkedIn.

We're continuing to iterate, with Neo4j optimizations, agent evaluation, and guardrails against prompt injection on the roadmap.
