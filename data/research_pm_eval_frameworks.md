# Raw Research Findings: PM-Centric LLM Evaluation

*Collected 2026-03-17 via web search across 25+ sources*

---

## THEME 1: PM-Specific Evaluation Frameworks

### The Core Shift: Product Evaluation vs Model Evaluation
When evaluating LLMs, PMs must decide whether they are evaluating the **base model** or the **product that wraps it** (prompts, retrieval, tools, guardrails, UI). Product evaluation asks the only question that matters: **Does the system solve the user's task accurately, safely, and consistently?** (Source: Lenny's Newsletter)

### Three-Stage PM Framework (Adaline Labs)
1. **Stage 1 — Map Real-World Performance Patterns**: Segment failures by persona, task, channel, context length. Build error taxonomies capturing refusal, hallucination, off-policy behaviors. Track cohorts through funnel analysis (prompt -> retrieval -> reasoning -> tools -> answer). Quantify impact by frequency AND business cost.
2. **Stage 2 — Create Meaningful Quality Systems**: Combine task metrics (exact match, F1, semantic similarity), policy checks (safety compliance, PII detection, tool-use constraints), and LLM-as-a-judge calibration (rubric scoring validated against human review on holdout sets).
3. **Stage 3 — Continuous Monitoring**: Automated canary and shadow testing on live traffic. Nightly regression suites on high-value slices. Alerts monitoring win-rate deltas, error rates, token costs, safety violations. Weekly reviews with prompt refinement, tool fixes, auto-rollback.

### The Eval Formula (Lenny's Newsletter)
Each strong LLM-based eval contains four components:
1. **Role-setting**: Prime the judge-LLM
2. **Context provision**: Include actual data from your application
3. **Goal clarity**: Define what the judge should measure; what success/failure looks like
4. **Terminology grounding**: Define terms precisely so the judge LLM understands context-specific meanings

### Productboard's 4-Part Framework
1. **Define "Good"**: Articulate what success looks like for your specific use case — beyond generic benchmarks
2. **Encode Judgment**: Convert qualitative definitions into measurable criteria engineering can automate
3. **Collaborate with Engineering**: Shared understanding of what evaluations reflect actual product expectations
4. **Iterate Continuously**: Evals must evolve as models change, use cases expand, user behavior shifts

### ProductManagement.ai's 5-Step Framework
1. Define success clearly (e.g., 80% issue resolution without escalation + 4/5 satisfaction rating)
2. Use both automated metrics and human checks
3. Establish a reliable baseline (rule-based systems, public models, benchmarks, custom datasets)
4. Pilot with real users (A/B testing)
5. Continuously monitor and iterate

### PM Competency Evolution (PromptLayer)
PMs are evolving into **"AI Orchestrators"** who navigate the "Iron Triangle" balancing:
- Latency vs accuracy
- Cost vs scale
- Generalization vs specialization

They need fluency in: training vs inference, performance evaluation methods, probabilistic thinking (outputs vary even with identical inputs), and ethical leadership on bias/privacy/fairness.

---

## THEME 2: What Metrics Do Leading AI Product Teams Track?

### Product School's 9-Dimension Taxonomy
1. **Accuracy & Core Performance**: Precision, recall, F1, log loss, ROC-AUC. "Starting line, not finish line" — use for model comparison and regression detection.
2. **Latency & Throughput**: Avg response time (ms), tail latency (p95/p99), requests/sec under load. Conversion drops when latency exceeds ~1 second.
3. **UX Trust & Consistency**: Consistency rate for identical inputs, determinism score, refusal quality for unsafe requests, clarity rating. "A perfectly correct answer that looks confusing still damages trust."
4. **Hallucination Rate & Factuality**: % of unverifiable/false outputs, citation coverage, faithfulness error rate vs source materials.
5. **Bias & Fairness**: Demographic parity, equalized odds, subgroup performance, toxicity scores. Disparity threshold: "no more than 1.2x between groups." (Apple Card case study cited.)
6. **Robustness & Safety**: Adversarial robustness, jailbreak success rate, policy compliance %, red-team testing.
7. **Cost & Efficiency**: Cost per query, tokens per output, throughput-per-dollar, cost per user action.
8. **Drift & Data Quality**: Population Stability Index (PSI >0.2 = warning), KL divergence, output pattern stability.
9. **Human Evaluation**: Rater agreement rate (target >80%), rater drift detection, scenario coverage, turnaround time.

### Confident AI's Metric Categories by Architecture

**RAG Metrics:**
- Faithfulness (claims aligned with retrieval context)
- Answer Relevancy (concisely addresses user input)
- Contextual Precision (relevant nodes ranked early)
- Contextual Recall (retrieved info supports expected output)
- Contextual Relevancy (proportion of relevant context)

**AI Agent Metrics:**
- Task Completion (end-to-end via LLM-as-judge)
- Tool Correctness (exact-match validation)
- Argument Correctness (tool parameters make semantic sense)
- Plan Quality (complete, logical plans)
- Plan Adherence (follows created plans)
- Step Efficiency (no unnecessary actions)

**Foundational Model Metrics:**
- Hallucination (reference-based or SelfCheckGPT)
- Toxicity (Detoxify or custom G-Eval rubrics)
- Bias (racial, gender, political — hardest to implement reliably)

**Use-Case Specific:**
- Helpfulness (subjective user value — G-Eval recommended)
- Prompt Alignment (follows instructions)
- Summarization (factual accuracy + completeness, score = min of two)

**Multi-Turn Conversational:**
- Turn Faithfulness, Turn Relevancy, Turn Contextual Precision/Recall/Relevancy

### The 5-Metric Rule (Confident AI)
Optimal evaluation = **1-2 custom metrics** (use-case specific) + **2-3 generic metrics** (architecture-specific). Prevents metric bloat while maintaining business relevance.

### Correction-to-Completion Ratio
Described as "the most reliable evaluation metric for production use cases" — measures accuracy and effectiveness in providing correct information or completing tasks. Consistently low ratios indicate genuinely useful LLMs.

---

## THEME 3: Evaluation Methods (How to Actually Score)

### Three Evaluation Approaches (Lenny's Newsletter)

**1. Human Evals**
- User feedback (thumbs up/down) or paid human labelers
- Strengths: Directly tied to end users
- Limitations: Sparse signals, unclear meaning, costly at scale

**2. Code-Based Evals**
- API call checks, code generation validation, format compliance
- Strengths: Fast and cheap; works for deterministic logic
- Limitations: Poor for subjective or open-ended tasks

**3. LLM-Based Evals (LLM-as-Judge)**
- External "judge" LLM grades outputs using natural language prompts
- Strengths: Highly scalable; PMs can write directly; generates explanations
- Limitations: Probabilistic results; needs initial setup + labeled examples

### Scoring Approaches (Confident AI)

**Statistical Scorers** (mostly outdated for LLM eval):
- BLEU: N-gram precision matching
- ROUGE: Recall-based n-gram overlap
- METEOR: Precision + recall with synonym matching
- Cannot capture semantic nuance; not recommended as primary metrics

**Model-Based Non-LLM Scorers:**
- NLI (entailment classification), BLEURT, BERTScore, MoverScore
- Less capable than LLMs; lower accuracy

**LLM-as-Judge Approaches:**
- **G-Eval**: Chain-of-thought eval steps, form-filling, 1-5 scale. Best for subjective criteria. Most reliable for custom metrics.
- **DAG**: Decision-tree with LLM nodes, deterministic score assignment. Best for clear success criteria.
- **QAG**: Extracts claims, converts to yes/no questions, checks against reference. Best for RAG/reference-dependent. Superior reliability for objective evaluation.
- **Prometheus**: Open-source LLM fine-tuned on GPT-4 feedback. Uses provided rubric. Comparable to GPT-4 performance.
- **SelfCheckGPT**: Sampling-based hallucination detection. Reference-less. Assumes hallucinations lack reproducibility.

### Standard Eval Criteria (commonly tracked)
- Hallucination: "Is the agent using provided context or making things up?"
- Toxicity/tone: Harmful or undesirable language
- Overall correctness: Primary goal achievement
- Code generation: Quality validation
- Summarization quality: Summary effectiveness
- Retrieval relevance: RAG system performance

---

## THEME 4: Trust, Safety, and User Satisfaction

### Trust Mechanics (AI Product Reliability Guide)
- Trust forms when **verification cost** (checking outputs) becomes lower than manual task completion
- **The first visible error creates a structural shift**: users move from delegation to supervision
- Once users adopt verification behavior, "reliability perception lags reliability reality" — even improvements don't immediately restore trust
- Users evaluate AI products on **"worst recent experience"** rather than average accuracy

### Trust Adoption Progression
Trial -> Assisted usage (active supervision) -> Workflow integration (delegation)
- Progression depends on **consistent behavior**
- Daily failure count matters more than percentage accuracy
- A 5% error rate = 10 daily failures for frequent users, reverting them to manual work

### Confidence Calibration (Multiple Sources)
- Calibration ensures that if a model claims 90% confidence, it should be correct ~90% of the time
- LLMs frequently suffer from **miscalibration** — trained to be persuasive, not truthful
- Methods: Temperature Scaling, Isotonic Regression, Multicalibration, CCPS (nudges to internal state)
- **Practical application**: Calibrated confidence scores enhance trustworthiness in customer service, content creation, educational platforms

### Real-Time Trust Scoring (Cleanlab TLM)
- Assigns real-time confidence scores to AI Agent responses
- Below threshold: suppress output, escalate to humans, or return safe fallbacks ("I'm not sure")
- Benchmark results with 0.8 trust threshold across agent architectures:
  - Act (zero-shot): 56.2% reduction in incorrect responses
  - ReAct (zero-shot): 55.8% reduction
  - ReAct (few-shot): 15.7% reduction
  - PlanAct: 24.5% reduction
  - PlanReAct: 10.0% reduction
- "Even nuanced inaccuracies that might slip past readers can be flagged by low confidence scores"
- Teams can tune thresholds to meet specific error tolerances (e.g., max 5% incorrect rate)

### UX Trust Signals (Product School)
- **Consistency rate** for identical inputs
- **Determinism score** measuring output variation
- **Refusal quality** for unsafe requests (graceful, helpful refusals vs hard blocks)
- **Clarity rating** for user comprehension

### User Satisfaction Metrics

**CSAT (Customer Satisfaction Score):**
- Measures satisfaction via feedback surveys/ratings (1-5 or 1-10 scales)
- Industry benchmark: "good" = 75-85%
- Targets: >=85% for general use, >=90% for high-stakes (pharmacy, wellness)
- Formula: satisfied responses / total responses x 100

**Thumbs Up/Down:**
- Simple per-message feedback
- Helps zero in on underperforming replies
- Dashboard: count of thumbs up/down per intent/topic

**Retention:**
- Customer satisfaction has direct link with retention and word-of-mouth
- CSAT 85-95% range significantly improves retention
- First Contact Resolution >70% helps cut support costs

**Net Promoter Score (NPS):**
- "Would you recommend?" quantification
- Usually deployed end-of-conversation

### Safety and Governance
- Red-team with "provocative or extreme prompts" to detect unsafe outputs
- Bias probes ensuring results don't skew by demographics
- Robust hallucination detection for high-stakes domains (medical, financial)
- Jailbreak success rate monitoring
- Policy compliance percentage tracking

### Harvey's BigLaw Bench (Real-World Example)
Legal AI evaluation framework with:
- Task-specific rubrics derived from actual billable lawyer work
- Multi-dimensional scoring (answer scores, source accuracy, hallucination penalties)
- Revealed general-purpose models struggle with source attribution while specialized models achieve higher quality

---

## THEME 5: Cost Efficiency Metrics

### Key Cost Metrics to Track
- **Cost per query**: Total API cost / number of queries
- **Tokens per query**: Average input + output tokens per request
- **Cache hit rate**: % of queries served from cache
- **Model usage mix**: Distribution across models (cheap vs expensive)
- **Failure and retry rates**: Wasted spend on failed calls
- **GPU utilization**: Infrastructure efficiency
- **Throughput-per-dollar**: Queries processed per dollar spent
- **Cost per user action**: Business outcome cost attribution

### Business-Outcome Cost Metrics
Connect technical costs to business value:
- Cost per resolved customer-support ticket
- Cost per processed document or contract
- Cost per sale closed with AI-assisted service
- Cost per generated article/report/marketing copy

### The Scale of the Problem
- Enterprise LLM spending doubled from $3.5B to $8.4B (late 2024 to mid-2025)
- Enterprise LLM market projected: $6.7B -> $71.1B by 2034
- 72% of companies plan to increase LLM spending
- Token pricing asymmetry: output tokens cost 3-5x more than input tokens

### Cost Optimization Strategies (with measured impact)

**Prompt Engineering**: 20-40% token reduction without quality loss
**Prompt Compression** (LLMLingua): Up to 20x compression preserving meaning
**Model Cascading**: Route 90% to cheaper models, escalate complex queries. ~87% cost reduction.
- GPT-4: ~$24.70/million tokens vs Mixtral 8x7B: ~$0.24/million tokens (100x spread)
- Routing 60% to cheaper models = 50%+ cost reduction
**Semantic Caching**: 31% of enterprise queries are semantically similar. 40-70% cost reduction, latency 850ms -> 120ms.
**Provider-Level Caching**: Anthropic prompt caching = 90% reduction for repeated long prompts. OpenAI auto-caching = 50% for identical requests.
**RAG**: 70% reduction in context-related token usage
**Model Distillation**: 50-85% cost reduction while maintaining quality
**Batch Processing**: Up to 90% overhead reduction
**Self-Hosting Threshold**: Cost-effective at ~1M monthly queries. Hardware ($10K-50K) recovers in 6-12 months.

### ROI Framework Example
- Pre-optimization: $10,000/month
- Post-optimization: $2,000/month (80% reduction)
- Implementation: 160 hours x $150/hr = $24,000
- **Payback: 3 months**

### Real-World Cost Reduction Example
$47,000/month -> $28,000/month (42% reduction) via:
- RouteLLM for support queries (40% are simple FAQ)
- Semantic caching for FAQ-style questions
- Budget alerts at 80% threshold

---

## THEME 6: PM vs Engineer Perspective — What "Good" Looks Like

### The Engineer's View of "Good"
- High accuracy on benchmarks (MMLU, HumanEval, HELM)
- Low perplexity, high BLEU/ROUGE scores
- Fast inference (low latency, high throughput)
- Efficient token usage
- Model performs well on leaderboard
- Code runs without errors at specified frequency
- Robust against adversarial inputs

### The PM's View of "Good"
- **Does the system solve the user's task?** (not: does it score well on benchmarks)
- Does it reduce time-to-value for the user?
- Do human agents still need to edit outputs? How often?
- Does it reduce average resolution time?
- Are customers getting faster/better outcomes?
- Is it performing consistently across different types of tasks?
- Is the AI performing consistently across user segments?
- Does the experience build trust or erode it?
- Can we ship it without embarrassment?

### Key Quote (Aman Khan, Arize AI)
"The reality is that LLMs hallucinate. Our job is to make sure that they don't embarrass us."

### The Gap Between Demo and Production
- Demos present curated prompts under ideal conditions
- Production reveals: unfiltered user input, noisy data, repeated daily interactions, varied latency
- Users evaluate on "worst recent experience" not average accuracy
- "Comparing traditional testing to LLM testing is like testing a spreadsheet versus testing a chef: the chef can make the same dish taste slightly different each time"

### Critical PM Insight (Product School)
"Evals are to AI PMs what A/B testing and product analytics were to digital PMs a decade ago." Traditional accuracy alone fails because high-performing models in labs still erode trust through slowness, bias, inconsistency, or factual errors in production.

---

## THEME 7: The Seven Layers of AI Product Reliability

### The System Beyond the Model (AI Product Reliability Guide)
Reliability depends on seven coordinated layers, NOT just the model:
1. **Intent Resolution** — interpreting ambiguous user requests
2. **Retrieval** — gathering contextual grounding data
3. **Prompt Construction** — assembling the actual specification
4. **Generation** — probabilistic token prediction
5. **Tool Execution** — function calls and external actions
6. **Guardrails** — policy filters and safety constraints
7. **Rendering** — presentation to users

The model is "only one component and often not the primary determinant" of reliability.

### Reliability > Features
- AI adoption follows: Trial -> Assisted usage -> Workflow integration (delegation)
- Progression depends on consistent behavior
- "Reliability is managed as a product property" — directly influences retention more than feature breadth

---

## THEME 8: Implementation Tiers and Governance

### Tiered Evaluation Approach (Product School)
- **Tier 0 (Smoke tests)**: Ultra-fast daily checks
- **Tier 1 (Core suite)**: Weekly runs on golden datasets, ~30-60 minutes
- **Tier 2 (Extended)**: Bi-weekly comprehensive with human review

### Context-Specific Priorities

**B2B/Enterprise:**
- Prioritize fairness testing and audit trails
- Require explanation completeness (>=90% valid citations)
- Maintain model cards documenting performance
- Focus on defensibility over speed

**B2C/Consumer:**
- Sub-second interaction target with p95 monitoring
- Measure perceived consistency and tone uniformity
- Hallucination rate <5% for general use
- In-product thumbs-up/down feedback

**Internal Tools:**
- Enforce deterministic responses to identical queries
- Validate against verified company policies
- Test refusal quality and data boundary respect
- Maintain "golden sets" of policy questions

### Ownership Structure
- Every metric needs one accountable owner
- PMs usually own the overall eval program
- Engineers/data scientists manage specific metric areas
- Weekly quality review cadence
- Monthly executive snapshot reporting trends
- Tie model rollouts to measurable performance shifts

### Getting Started (Productboard's 5 Steps)
1. Identify failure modes — document where your AI currently breaks
2. Define evaluation criteria — judgment-based quality rubrics
3. Build baseline tests — start with 10-20 representative examples
4. Automate assessment — implement automated + human-in-the-loop
5. Monitor and adapt — continuously refine from production learnings

---

## THEME 9: Real-World Failure Stories and Cautionary Tales

### Meta's Galactica
- Hallucinated references and spewed toxic content
- Sounded authoritative despite being wrong
- Meta pulled the demo within 3 days
- Lesson: Formal evaluation prevents shipping features that "feel" polished but contain subtle flaws

### Apple Card Gender Bias
- Algorithm assigned significantly lower credit limits to women
- A fairness failure despite overall accuracy
- Lesson: Subgroup performance testing is non-negotiable

### Industry-Wide Failure Rates
- Only 48% of pilot AI applications graduate to production
- 30% of generative AI projects abandoned after proof-of-concept
- MIT reports 95% pilot failure rates
- Root causes: inadequate evaluation frameworks, weak governance, poor integration

---

## Sources

- [A PM's Complete Guide to Evals — Lenny's Newsletter](https://www.lennysnewsletter.com/p/beyond-vibe-checks-a-pms-complete)
- [LLM Evals Are Every PM's Secret Weapon — Adaline Labs](https://labs.adaline.ai/p/llm-evals-are-product-managers-secret-weapon)
- [Guide to Evaluating LLM-Powered Products — ProductManagement.ai](https://www.productmanagement.ai/p/a-guide-to-evaluating-llm-powered)
- [Evaluation Metrics for AI Products That Drive Trust — Product School](https://productschool.com/blog/artificial-intelligence/evaluation-metrics)
- [AI Evals for Product Managers — Productboard](https://www.productboard.com/blog/ai-evals-for-product-managers/)
- [AI Product Reliability Guide — Prayerson](https://www.iamprayerson.com/p/ai-product-reliability-a-guide-for-product-managers)
- [PM Levels LLM Competency — PromptLayer](https://blog.promptlayer.com/product-manager-levels-llm-competency-the-new-rules-of-ai-product-management/)
- [LLM Evaluation Metrics — Confident AI](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [How to Cut LLM Costs with Metering — Pluralsight](https://www.pluralsight.com/resources/blog/ai-and-data/how-cut-llm-costs-with-metering)
- [LLM Cost Optimization Guide — Koombea](https://ai.koombea.com/blog/llm-cost-optimization)
- [How to Implement Effective AI Evaluations — Mind the Product](https://www.mindtheproduct.com/how-to-implement-effective-ai-evaluations/)
- [Trust Scoring Benchmarking — Cleanlab](https://cleanlab.ai/blog/agent-tlm-hallucination-benchmarking/)
- [AI Evaluation Metrics: CSAT — Francesca Tabor](https://www.francescatabor.com/articles/2025/7/10/ai-evaluation-metrics-user-satisfaction-csat)
- [LLM Evaluation in 2025 — Medium/QuarkAndCode](https://medium.com/@QuarkAndCode/llm-evaluation-in-2025-metrics-rag-llm-as-judge-best-practices-ad2872cfa7cb)
- [AI Product Manager — Arize AI](https://arize.com/ai-product-manager/)
- [LLM Evaluation Guide — Evidently AI](https://www.evidentlyai.com/llm-guide/llm-evaluation)
- [Multicalibration for Confidence Scoring — arXiv](https://arxiv.org/pdf/2404.04689)
- [Trust Calibration Maturity Model — arXiv](https://arxiv.org/pdf/2503.15511)
- [Why AI Evaluation Is a Must-Have Skill — Product School](https://productschool.com/blog/artificial-intelligence/ai-evals-product-managers)
- [AI Evaluation Revolution — Aakash Gupta / Medium](https://aakashgupta.medium.com/the-ai-evaluation-revolution-why-every-product-manager-must-master-this-critical-skill-in-2025-0458c4ac6097)
- [LLM Benchmarks — Evidently AI](https://www.evidentlyai.com/llm-guide/llm-benchmarks)
- [Confidence Scores in LLMs — Infrrd](https://www.infrrd.ai/blog/confidence-scores-in-llms)
- [Chatbot Analytics KPIs — Quickchat AI](https://quickchat.ai/post/chatbot-analytics)
- [LLM Cost Tracking — TrueFoundry](https://www.truefoundry.com/blog/llm-cost-tracking-solution)
