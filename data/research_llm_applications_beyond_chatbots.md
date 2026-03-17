# Research: LLM Applications Beyond Chatbots & Their Quality Challenges

> Raw findings from web research (March 2026). Evidence collection only -- no metric definitions yet.

---

## 1. CODE GENERATION

### What It Is
LLMs generate code from natural language prompts, autocomplete functions, fix bugs, translate between languages, and write tests. Production use cases include GitHub Copilot, Cursor, Amazon CodeWhisperer, and internal enterprise code assistants.

### Unique Quality Challenges

**Functional Correctness vs. Security Gap**
- A ~25-35% absolute gap exists between functionally correct code and simultaneously secure + correct code. GPT-4o yields func@10 = 90.7% but func-sec@10 = only 65.3% (CWEval benchmark).
- Secure-pass@1 rates remain <12% for all models tested, even when functional pass@1 exceeds 50%.
- 12-65% of generated code snippets are non-compliant with basic secure coding standards or trigger CWE-classified vulnerabilities.

**Defect Distribution**
- Consistent across models: ~90-93% code smells, 5-8% bugs, ~2% security vulnerabilities.
- LLM code increases overall smell rates by 63.34%, with largest growth in implementation smells (73.35%), driving long-term maintainability concerns.
- No correlation exists between a model's functional benchmark performance and code quality/security.

**Multi-Turn Degradation**
- MT-Sec benchmark shows a consistent 20-27% drop in "correct and secure" outputs from single-turn to multi-turn coding scenarios.

**Key Metrics Used**
- pass@k (functional correctness via unit tests)
- func-sec@k (joint functional correctness + security)
- SAFE@k (partial credit where security doesn't ruin utility)
- Static analysis defect counts (code smells, bugs, vulnerabilities)
- Benchmarks: HumanEval, MBPP, APPS, HumanEval-X/XL, MultiPL-E

**How Evaluation Differs from Chatbots**
- Requires executable, testable output -- not just fluent text
- Security is a first-class concern (not relevant for chatbots)
- Maintainability and code smell density matter for production
- Multi-turn context degrades quality significantly

---

## 2. RAG (RETRIEVAL-AUGMENTED GENERATION)

### What It Is
LLMs augmented with external knowledge retrieval to answer questions, power enterprise search, knowledge bases, and customer support. The retriever fetches context; the generator produces answers grounded in that context.

### Unique Quality Challenges

**Retriever Quality**
- Bad retrieval = bad answers, regardless of generator quality
- Chunk size and top-K configuration directly impact output
- Ranking order matters -- relevant context buried low gets ignored

**Faithfulness / Hallucination**
- Most common RAG hallucination is NOT random fabrication but unfaithfulness to the retrieved context
- Model may contradict, misinterpret, or selectively ignore retrieved passages

**Context Window Limitations**
- Too much irrelevant context dilutes signal
- Too little context misses critical information

**Key Metrics Used**

*Retriever Metrics:*
- Contextual Relevancy: proportion of retrieved chunks that are relevant to the input
- Contextual Recall: proportion of ground-truth facts attributable to retrieved chunks
- Contextual Precision: whether relevant chunks are ranked higher than irrelevant ones
- NDCG (Normalized Discounted Cumulative Gain): accounts for both relevance and ranking position; correlates more strongly with end-to-end RAG quality than binary relevance metrics

*Generator Metrics:*
- Answer Relevancy: how relevant the generated response is to the given input
- Faithfulness: proportion of claims in the output not contradicted by retrieval context (hallucination rate)
- Answer Correctness: factual accuracy compared to ground truth

*Evaluation Methods:*
- LLM-as-a-Judge (GPT-4 matched human judgment ~80% of the time)
- QAG (Question Answer Generation) scoring
- Ragas, TruLens, DeepEval frameworks

**How Evaluation Differs from Chatbots**
- Two-component system (retriever + generator) requires separate evaluation
- Faithfulness to source context is primary concern (not creativity)
- Retrieval quality is a hard prerequisite -- poor retrieval cannot be compensated by a better generator
- Production best practice: CI/CD gates with automated evals

---

## 3. AGENTIC AI / TOOL-CALLING SYSTEMS

### What It Is
LLMs that reason, plan, select tools, execute multi-step workflows, and adapt to changing contexts. Examples: coding agents (Devin, Claude Code), customer service agents, data analysis pipelines, autonomous workflows.

### Unique Quality Challenges

**Tool Selection & Parameter Accuracy**
- Must correctly decide IF a tool is needed, WHICH tool, and with WHAT parameters
- Parameter hallucination: fabricating values not present in user query
- Type checking (integers, floats, strings, booleans) and range validation critical

**Multi-Step Reasoning & State Management**
- Tracking preconditions and postconditions across sequential calls
- Managing dependencies where one tool's output feeds into another
- Validating final system state matches ground truth
- 20-27% degradation in multi-turn scenarios

**Task Completion vs. Safety Paradox**
- Agents can achieve 100% task completion while exhibiting only 33% policy adherence
- "Perfect tool sequencing but ignoring safety guidelines" -- invisible to outcome-only metrics

**Error Recovery**
- Graceful handling of failed tool calls
- Avoidance of invalid state transitions ("minefields")
- Recovery without cascading failures

**Non-Determinism**
- Same input can produce different tool-calling sequences
- Consistency across invocations is a unique challenge

**Key Metrics Used**
- Tool Correctness: whether correct tools were called (deterministic, supports order/frequency flexibility)
- Tool-Calling Efficiency: whether tools were used in most efficient way (redundant calls)
- Task Completion: end-to-end success rate
- Parameter Accuracy: correct types, values, required vs. optional handling
- Progress Rate: correct function calls until first error
- Consistency: stable results across invocations
- Policy Adherence: safety compliance during execution
- Milestone Similarity (0-1): partial progress scoring
- AST Correctness: syntax validation for function calls

*Benchmarks:*
- Berkeley Function-Calling Leaderboard (BFCL v1/v3)
- ToolSandbox, ToolTalk, HammerBench, ComplexFuncBench
- tau-Bench (state matching, pass^k consistency)

*Evaluation Framework (4-Pillar):*
1. LLM: instruction following, safety alignment
2. Memory: storage correctness, retrieval accuracy
3. Tools: selection, parameter mapping, sequencing, error interpretation
4. Environment: resource limits, operational guardrails

**How Evaluation Differs from Chatbots**
- Actions have real-world consequences (API calls, database writes)
- Binary outcome metrics miss critical behavioral failures
- Must evaluate the TRAJECTORY (how it got there), not just the final answer
- Safety/policy compliance is a parallel evaluation axis
- Multi-agent coordination adds complexity

---

## 4. SUMMARIZATION

### What It Is
LLMs condensing long documents, articles, reports, meeting transcripts, medical records, or legal documents into shorter, accurate summaries.

### Unique Quality Challenges

**Faithfulness / Hallucination**
- Models may inject plausible-sounding facts not in the source
- Faithfulness metrics capture hallucinations to some extent but are not foolproof
- Faithfulness evaluation cannot rely solely on LLM-as-judge -- hybrid approaches needed

**Compression vs. Completeness Trade-off**
- Higher compression risks omitting critical information
- Lower compression may fail to meaningfully summarize

**Extractive vs. Abstractive**
- Traditional metrics (BLEU, ROUGE) designed for extractive summarization
- LLM-generated abstractive summaries paraphrase, making n-gram overlap metrics unreliable
- BERTScore helps but still misses semantic nuances

**Key Metrics Used**
- Accuracy: ROUGE, BLEU, METEOR, BERTScore (how closely output resembles expected answer)
- Faithfulness: SummaC (sentence-level alignment), QAFactEval (QA-based consistency)
- Compression: ratio of tokens in summary vs. source
- Extractiveness/Coverage: how much summary text comes directly from source
- Density: quantifies word sequence extraction patterns
- Efficiency: computational resources for inference

*Evaluation Methods:*
- LLM-as-a-Judge: GPT-4 matched human judgment ~80% of the time; Spearman rho ~0.67 on answer correctness
- QAG (Question Answer Generation) scoring
- Human evaluation remains "gold standard"
- AI red-teaming for adversarial testing

**How Evaluation Differs from Chatbots**
- Source document is the ground truth (not general knowledge)
- Faithfulness to source is the primary quality dimension
- Compression ratio is a unique metric not applicable to chatbots
- Traditional NLP metrics (ROUGE, BLEU) are partially applicable but insufficient for abstractive summaries

---

## 5. DATA EXTRACTION / STRUCTURED OUTPUT

### What It Is
LLMs extracting structured data from unstructured text: parsing invoices, medical records, legal contracts, resumes, tables, forms. Output is JSON, database records, or structured schemas.

### Unique Quality Challenges

**Schema Compliance**
- Output must conform to exact schemas (types, required fields, enums)
- Minor format errors break downstream pipelines

**Field-Level vs. Record-Level Accuracy**
- A record with 9/10 correct fields may still be useless if the wrong field is critical
- Need both field accuracy and output accuracy metrics

**Confidence Calibration**
- Using trust scoring to flag problematic outputs: focus human review on 1-5% of untrustworthy cases
- Enables 95-99% of work to be LLM-automated

**Multi-Dimensional Quality (especially in regulated domains)**
- Task accuracy, structural quality, human-in-the-loop burden, stability/reliability, regulatory compliance
- Conventional quality methods fail to assess GenAI extraction systems

**Key Metrics Used**
- Field Accuracy: proportion of individual fields extracted correctly
- Output Accuracy: proportion of samples where EVERY field is correct
- F1 Score: balancing precision and recall at entity level
- Tree Edit Distance: structural accuracy for hierarchical data
- Trust Scores: confidence-based flagging for human review
- Schema Compliance Rate: percentage of outputs matching expected schema

**How Evaluation Differs from Chatbots**
- Output must be machine-parseable, not human-readable prose
- Schema compliance is binary (valid or invalid)
- Partial correctness has different implications (9/10 fields correct may still fail)
- Confidence calibration enables production automation at scale

---

## 6. CONTENT GENERATION (Marketing, Product Descriptions, Reports)

### What It Is
LLMs generating marketing copy, product descriptions, blog posts, social media content, reports, and documentation at scale.

### Unique Quality Challenges

**Quality Control at Scale**
- Gen AI's tendency to "make things up, leave things out, and create so many possibilities" (HBR)
- Human reviews are expensive and can handle only a fraction of total output
- Amazon's Catalog AI initially had 80% failure rate

**Three Failure Modes (HBR/Amazon)**
1. Hallucination: fabricating specifications (e.g., claiming 15 horsepower without source data)
2. Omission: leaving out relevant details
3. Over-generation: too many possibilities, hard to determine effectiveness

**Brand Voice & Consistency**
- Maintaining consistent tone, style, and terminology across thousands of outputs
- Template compliance and format adherence

**Evaluation-Action Gap**
- Even after reliability checks, ~40% of content improves metrics while 60% performs negatively
- A/B testing is essential -- automated quality checks alone are insufficient

**Key Quality Control Methods (Amazon's 4-Step)**
1. Baseline Auditing: human auditors compare AI output against known information
2. Multi-Layer Guardrails: rule enforcement + statistical profile controls + AI-checking-AI
3. A/B Testing Integration: actual customer response measurement
4. Continuous Learning: feedback loops for autonomous improvement

**How Evaluation Differs from Chatbots**
- Must evaluate business impact (conversion, engagement), not just text quality
- Brand voice consistency across thousands of outputs
- A/B testing required -- subjective quality alone is insufficient
- Scale creates evaluation bottlenecks (can't human-review everything)

---

## 7. TEXT CLASSIFICATION & ENTITY RECOGNITION

### What It Is
LLMs classifying text into categories (sentiment, intent, topic, urgency) and extracting named entities (people, organizations, dates, amounts) from unstructured text.

### Unique Quality Challenges

**Class Imbalance**
- Rare categories are harder to classify correctly
- F1 score preferable to raw accuracy for imbalanced datasets

**Label Consistency**
- Same text classified differently across runs (non-determinism)
- Domain boundary precision (when does "fintech" become "banking"?)

**Taxonomy Adherence**
- Must output only valid labels from a predefined set
- Hallucinated categories are a unique failure mode

**Performance Drift**
- Monitoring F1, precision, recall across deployments and time
- Input data evolution causes gradual degradation

**Key Metrics Used**
- Precision: percentage of predicted positives that are actually correct
- Recall: percentage of actual positives that are captured
- F1 Score: harmonic mean of precision and recall
- Accuracy: overall correctness (misleading with imbalanced classes)
- Per-class metrics: precision/recall/F1 broken down by category
- Production telecom example: F1 of 0.83 on benchmarked answers

**How Evaluation Differs from Chatbots**
- Evaluation is deterministic (correct label or not) vs. subjective quality
- Standard ML metrics (precision, recall, F1) directly applicable
- Class distribution matters significantly
- Taxonomy compliance is binary

---

## 8. TRANSLATION

### What It Is
LLMs translating text between languages, including technical documentation, marketing content, legal documents, and real-time communication.

### Unique Quality Challenges

**Fluent but Unfaithful**
- LLM translations can be extremely fluent while being hallucinated or unfaithful to source
- COMET assigns overly generous scores to confident but unfaithful translations (training data gap)

**Cultural Context**
- Preserving cultural nuances, idioms, technical terminology
- Tone adaptation across cultural contexts

**Metric Limitations**
- BLEU relies on n-gram overlap -- poor for paraphrased translations
- COMET better correlated with human judgment but has known failure modes with LLM-generated translations
- Neither metric reliably catches hallucinated translations

**Key Metrics Used**
- BLEU: n-gram overlap (score 0-1, >0.3 considered good; low correlation with human judgment)
- COMET: neural metric, more resilient to word order/synonyms/paraphrasing; better human correlation
- METEOR: includes synonyms in overlap calculation
- TER (Translation Edit Rate): minimum edits needed
- Human evaluation: WMT official rankings still based on human scores
- Hybrid model: automated scoring for speed + expert human reviewers for edge cases

**How Evaluation Differs from Chatbots**
- Source text provides objective reference (unlike open-ended chat)
- Semantic equivalence across languages is the core metric
- Fluency can mask unfaithfulness (unique to translation)
- Domain-specific terminology accuracy is critical

---

## 9. SEARCH & INFORMATION RETRIEVAL (Semantic Search, Enterprise Search)

### What It Is
LLM-powered search that understands intent, provides personalized results, and dynamically adapts to user behavior. Used in e-commerce, knowledge bases, enterprise content systems.

### Unique Quality Challenges

**Intent Disambiguation**
- Understanding what the user actually wants vs. what they typed
- Multi-intent queries

**Ranking Quality**
- Relevant results buried in noise
- Personalization vs. objectivity trade-off

**Cross-Source Synthesis**
- Combining information from multiple sources coherently
- Maintaining source attribution

**Key Metrics Used**
- Precision@K: proportion of top-K results that are relevant
- Recall@K: proportion of relevant results captured in top-K
- NDCG: normalized discounted cumulative gain (ranking-aware)
- MRR (Mean Reciprocal Rank): position of first relevant result
- Click-through rate / user engagement metrics

**How Evaluation Differs from Chatbots**
- Ranking quality is the core metric (not applicable to chatbots)
- Requires retrieval-specific metrics (NDCG, MRR, Precision@K)
- User behavior signals (clicks, dwell time) are evaluation inputs
- Personalization quality adds a dimension

---

## 10. SYNTHETIC DATA GENERATION

### What It Is
LLMs generating training data, test cases, simulated datasets for ML model training, testing, and augmentation.

### Unique Quality Challenges

**Distribution Fidelity**
- Generated data must match real-world statistical distributions
- Mode collapse: generating repetitive, non-diverse data

**Privacy Compliance**
- Must not leak real data patterns or PII
- Differential privacy considerations

**Downstream Utility**
- Quality measured by performance of models trained ON the synthetic data
- Not just surface-level similarity

**Key Metrics**
- Statistical distribution comparison (KL divergence, Wasserstein distance)
- Downstream model performance (accuracy of models trained on synthetic data)
- Diversity metrics
- Privacy compliance audits

---

## CROSS-CUTTING THEMES

### Hallucination Is Universal But Manifests Differently
- **Chatbots**: fabricated facts in conversation
- **RAG**: unfaithfulness to retrieved context
- **Code Gen**: security vulnerabilities, fabricated APIs
- **Summarization**: injected facts not in source
- **Translation**: fluent but unfaithful translations
- **Extraction**: fabricated field values
- **Content Gen**: fabricated product specifications

### Evaluation Cannot Be One-Size-Fits-All
- Traditional NLP metrics (BLEU, ROUGE) designed for pre-LLM era; insufficient for abstractive/generative tasks
- LLM-as-a-Judge correlates ~80% with human judgment but has known blind spots
- Every application type needs domain-specific metrics PLUS common quality dimensions
- Human evaluation remains the gold standard but doesn't scale

### The Production Gap
- Amazon's experience: quality jumped from 20% to 80% pass rate through systematic refinement, yet "still a work in progress"
- Industry losses from hallucination-related incidents exceed $250M annually
- Hybrid approaches (automated + human) are the 2025 consensus
- CI/CD gates with automated evals becoming best practice for production RAG

### Multi-Dimensional Evaluation is the 2025 Norm
- Single-number accuracy is insufficient
- Composite panels: task accuracy + structural quality + human-in-the-loop burden + stability + compliance
- Behavioral assessment (how it got there) vs. outcome assessment (what it produced)

---

## SOURCES

### LLM Applications & Use Cases
- [Best LLM Use Cases for Business Growth 2026](https://citrusbug.com/blog/llm-use-cases/)
- [Definitive Guide to LLM Use Cases 2025](https://www.goml.io/blog/definitive-guide-to-llm-use-cases)
- [Top 7 LLM Use Cases 2026](https://context-clue.com/blog/large-language-models-llm-use-cases-in-2026/)
- [State of LLMs 2025 - Sebastian Raschka](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [AssemblyAI: LLM Use Cases 2026](https://www.assemblyai.com/blog/llm-use-cases)

### RAG Evaluation
- [Confident AI: RAG Evaluation Metrics](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [FutureAGI: RAG Evaluation Metrics Guide 2025](https://futureagi.com/blogs/rag-evaluation-metrics-2025)
- [Meilisearch: RAG Evaluation](https://www.meilisearch.com/blog/rag-evaluation)
- [GetMaxim: Complete Guide to RAG Evaluation 2025](https://www.getmaxim.ai/articles/complete-guide-to-rag-evaluation-metrics-methods-and-best-practices-for-2025/)
- [Evidently AI: RAG Evaluation Guide](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Tweag: Evaluating the Evaluators](https://www.tweag.io/blog/2025-02-27-rag-evaluation/)
- [Patronus AI: RAG Evaluation Metrics](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)
- [Ragas: Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

### Agent & Tool-Calling Evaluation
- [Confident AI: LLM Agent Evaluation Guide](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [QuotientAI: Evaluating Tool Calling - Literature Review](https://blog.quotientai.co/evaluating-tool-calling-capabilities-in-large-language-models-a-literature-review/)
- [Deepchecks: LLM Agent Evaluation](https://deepchecks.com/llm-agent-evaluation/)
- [arxiv: Beyond Task Completion Framework](https://arxiv.org/abs/2512.12791)
- [arxiv: Evaluation and Benchmarking of LLM Agents Survey](https://arxiv.org/abs/2507.21504)
- [DeepEval: Tool Correctness Metric](https://deepeval.com/docs/metrics-tool-correctness)
- [o-mega: Best AI Agent Benchmarks 2025](https://o-mega.ai/articles/the-best-ai-agent-evals-and-benchmarks-full-2025-guide)
- [AWS: Evaluating AI Agents at Amazon](https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-real-world-lessons-from-building-agentic-systems-at-amazon/)
- [Fluid AI: Rethinking LLM Benchmarks for Agentic AI](https://www.fluid.ai/blog/rethinking-llm-benchmarks-for-2025)

### Content Generation Quality
- [HBR: Addressing Gen AI's Quality-Control Problem](https://hbr.org/2025/09/addressing-gen-ais-quality-control-problem)
- [Koanthic: AI Content Quality Control Guide 2026](https://koanthic.com/en/ai-content-quality-control-complete-guide-for-2026-2/)
- [ScienceDirect: Measuring Quality of GenAI Systems](https://www.sciencedirect.com/science/article/pii/S0950584925001417)
- [Rellify: Quality Control in AI Content](https://www.rellify.com/blog/quality-control)
- [KODA: Evaluators for AI-Generated Responses](https://usekoda.com/blog/evaluators-quality-control-of-ai-responses/)

### Summarization Evaluation
- [CMU SEI: Evaluating LLMs for Text Summarization](https://www.sei.cmu.edu/blog/evaluating-llms-for-text-summarization-introduction/)
- [Confident AI: LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [DeepEval: Summarization Metric](https://deepeval.com/docs/metrics-summarization)
- [arxiv: Comprehensive Survey of Faithfulness Evaluation](https://acl-bg.org/proceedings/2025/RANLP%202025/pdf/2025.ranlp-1.74.pdf)
- [arxiv: Faithfulness Metric Fusion](https://arxiv.org/pdf/2512.05700)

### Code Generation Evaluation
- [EmergentMind: LLM Code Evaluation](https://www.emergentmind.com/topics/llm-generated-code-evaluation)
- [arxiv: Quality and Security of AI-Generated Code](https://arxiv.org/abs/2508.14727)
- [arxiv: Rethinking Secure Code Generation Evaluation](https://arxiv.org/pdf/2503.15554)
- [arxiv: Benchmarking Correctness and Security in Multi-Turn Code Gen](https://arxiv.org/abs/2510.13859)
- [EmergentMind: LLM Code Security](https://www.emergentmind.com/topics/security-of-llm-generated-code)

### Data Extraction & Structured Output
- [Cleanlab: Structured Output Benchmarks](https://cleanlab.ai/blog/structured-output-benchmark/)
- [arxiv: StructEval Benchmark](https://arxiv.org/html/2505.20139v1)
- [PMC: LLMs for Clinical Data Extraction](https://pmc.ncbi.nlm.nih.gov/articles/PMC12932350/)
- [Cleanlab: Real-Time Error Detection for Structured Outputs](https://cleanlab.ai/blog/tlm-structured-outputs-benchmark/)

### Hallucination Detection
- [Lakera: Guide to LLM Hallucinations](https://www.lakera.ai/blog/guide-to-hallucinations-in-large-language-models)
- [Nature: Detecting Hallucinations Using Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0)
- [vLLM: Token-Level Hallucination Detection](https://blog.vllm.ai/2025/12/14/halugate.html)
- [arxiv: Comprehensive Survey of LLM Hallucination](https://arxiv.org/abs/2510.06265)

### Translation Evaluation
- [Translated: MT Quality Evaluation in Age of LLM MT](https://translated.com/mt-quality-evaluation-in-the-age-of-llm-based-mt)
- [Pangeanic: Why BLEU is Not Enough](https://blog.pangeanic.com/evaluating-enterprise-mt-why-bleu-is-not-enough-and-how-comet-improves-quality-assessment)
- [International Achievers Group: Metrics for MT Evaluation](https://internationalachieversgroup.com/localisation/metrics-for-evaluating-machine-translation-using-llms/)

### Classification
- [Confident AI: LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [AIMultiple: LLM Evaluation Metrics & Methods](https://research.aimultiple.com/large-language-model-evaluation/)
- [Analytics Vidhya: Top 15 LLM Evaluation Metrics 2026](https://www.analyticsvidhya.com/blog/2025/03/llm-evaluation-metrics/)
