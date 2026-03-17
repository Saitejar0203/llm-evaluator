# LLM Evaluator for Product Managers

Evaluates LLMs on metrics that matter for product decisions: hallucination resistance, faithfulness, abstention quality, tool calling efficiency, and accuracy. Uses a Judge LLM to score responses with category-aware rubrics.

Built on top of [gauravvij/llm-evaluator](https://github.com/gauravvij/llm-evaluator), rewritten with PM-centric evaluation metrics and multi-turn tool calling.

## How It Works

You describe your task in plain English (e.g., "Customer support chatbot for a fintech lending app"). The tool then:

1. **Generates a knowledge base** from your task description using Gemini 3 Flash
2. **Creates a test suite** across 6 categories: in-context, out-of-context, general knowledge, multi-fact, edge case, and off-topic
3. **Runs all candidate models** through multi-turn tool calling (each model can search the knowledge base up to 3 times per question)
4. **Scores every response** using Gemini 3.1 Pro as the Judge LLM
5. **Ranks models** with strengths, weaknesses, cost, and latency breakdown

## Evaluation Metrics

### Judge-Scored (by Gemini 3.1 Pro)

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| Hallucination Resistance | 25% | Does the model make up facts? |
| Faithfulness | 20% | Does it stick to what the knowledge base says? |
| Abstention Quality | 20% | Does it say "I don't know" when it should? |
| Tool Calling Efficiency | 20% | Does it search the KB when needed, skip it when not? |
| Accuracy | 15% | Is the answer correct and complete? |

Each metric is scored differently depending on the test category. For example, hallucination on an in-context question (where the answer exists in the KB) is evaluated differently than on an out-of-context question (where the model should abstain).

### Computed from Response Data

- **Confidence Calibration** - gap between model's self-rated confidence and actual accuracy
- **Consistency** - same answer when asked the same question 3 times (optional)
- **Token Efficiency** - quality score per token spent
- **Quality-Adjusted Cost** - cost normalized by quality
- **Cost per Question** - actual dollar cost per answer including tool calls

## Architecture

```
Task Description
      |
      v
[Gemini 3 Flash] --> Knowledge Base (dict) + System Prompt + Test Suite
                                                    |
                                                    v
                                    [6 Candidate Models via OpenRouter]
                                    Multi-turn tool calling (up to 3 calls)
                                    search_knowledge_base(query) tool
                                                    |
                                                    v
                                    [Gemini 3.1 Pro - Judge]
                                    Category-aware rubrics
                                    5 metrics per response
                                                    |
                                                    v
                                    Ranked Results + Cost Analysis
```

**Generator:** Gemini 3 Flash ($0.50/M tokens) - creates knowledge base, system prompt, test suite

**Judge:** Gemini 3.1 Pro ($2.00/M tokens) - scores all responses with structured rubrics

**Candidates (default):** 6 models at similar price points (~$0.20-0.26/M input tokens):
- google/gemini-3.1-flash-lite
- openai/gpt-5-mini
- qwen/qwen3.5-122b-a10b
- minimax/minimax-m2.5
- inception/mercury-2
- mistralai/ministral-14b-2512

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Saitejar0203/llm-evaluator.git
cd llm-evaluator
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Run

```bash
# Interactive mode (prompts for task, number of tests, etc.)
python main.py

# CLI mode
python main.py --task "Customer support chatbot for a fintech app"
python main.py --task "Technical documentation Q&A bot" --num-tests 15
python main.py --task "HR knowledge assistant" --consistency
```

### CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--task` | `-t` | *(prompted)* | Task description |
| `--num-tests` | `-n` | `10` | Number of test cases (minimum 6) |
| `--output-dir` | `-o` | `./results` | Directory for markdown report |
| `--no-save` | | `False` | Skip saving report |
| `--consistency` | | `False` | Run consistency check (3 runs x 3 questions per model) |

## Test Categories

| Category | Share | Purpose |
|----------|-------|---------|
| In-Context | 25% | Answer exists in the knowledge base |
| Out-of-Context | 25% | Answer is NOT in the knowledge base (model should abstain) |
| Off-Topic | 15% | Completely unrelated to the task (tests role adherence) |
| General Knowledge | 15% | Common knowledge, no KB needed |
| Multi-Fact | 10% | Requires combining multiple KB sections |
| Edge Case | 10% | Ambiguous or tricky questions |

## Project Structure

```
llm-evaluator/
├── main.py                  # CLI entry point (7-step pipeline)
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py            # Models, pricing, evaluation dimensions
│   ├── openrouter_client.py # OpenRouter API client (Generator + Judge)
│   ├── suite_generator.py   # Knowledge base, system prompt, test suite generation
│   ├── knowledge_base.py    # Tool schema + search_knowledge_base function
│   ├── model_discovery.py   # Candidate model metadata from OpenRouter
│   ├── benchmarker.py       # Parallel benchmarking with multi-turn tool calling
│   ├── evaluator.py         # Judge evaluation with category-aware rubrics
│   ├── consistency.py       # Optional consistency checker
│   ├── schemas.py           # Pydantic models for test cases and scores
│   └── reporter.py          # Rich terminal output + markdown report
├── smoke_test.py            # Quick validation (2 models x 2 tests)
├── test_implementation.py   # Unit tests (64 tests)
├── results/                 # Generated evaluation reports
└── data/                    # Research documents
```

## Sample Results

After running an evaluation, you get:
- A ranked list of models with scores, strengths, and weaknesses
- A comparison table across all metrics
- Cost per question breakdown
- A detailed markdown report saved to `results/`

Each report includes metric definitions so the results are self-explanatory.

## License

MIT License

## Credits

Originally forked from [gauravvij/llm-evaluator](https://github.com/gauravvij/llm-evaluator). Rewritten with PM-centric evaluation metrics, multi-turn tool calling, category-aware rubrics, and computed metrics (calibration, consistency, efficiency).
