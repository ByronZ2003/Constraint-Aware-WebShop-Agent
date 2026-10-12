# Constraint-Aware WebShop

Clean research release of the Constraint-Aware WebShop agent.

## Main components

The framework contains:

1. Constraint Manager
2. Query Gate
3. Candidate Selector
4. Product Gate

## Candidate Selector

The Candidate Selector uses repeat-only intervention.

The Base Agent's first product click is preserved.

If the Base Agent later proposes a product that has already been visited,
the Candidate Selector:

1. extracts currently visible unvisited products;
2. builds a schema-aligned profile for every candidate;
3. computes product-type and requirement scores;
4. excludes candidates with a hard contradiction;
5. ranks the remaining candidates by overall Candidate score;
6. directly selects the highest-scoring unseen candidate.

There is NO minimum Candidate score threshold.

If no safe unseen candidate remains on the current page, the agent may
continue to the next search-results page.

## Candidate score

For candidate c:

    Score(c) =
        w_type * TypeScore(c)
        +
        w_req * RequirementScore(c)

Evidence-state mapping:

    SUPPORTED           = 1.00
    LIKELY_SUPPORTED    = 0.75
    UNKNOWN             = 0.35
    CONTRADICTED        = 0.00

## Online Candidate weights

The Candidate expert-weight implementation is located at:

    gates/candidate_online_weights.py

Adaptive training example:

    CANDIDATE_INIT_TYPE=0.50
    CANDIDATE_INIT_REQ=0.50
    CANDIDATE_ONLINE_ETA=0.10

Frozen evaluation example:

    CANDIDATE_INIT_TYPE=0.75
    CANDIDATE_INIT_REQ=0.25
    CANDIDATE_ONLINE_ETA=0.0

## Evaluation data

The repository contains the WebShop-small 1000-product setup and the
current 415-task evaluation split.

Main files:

    data/items_shuffle_1000.json
    data/items_ins_v2_1000.json
    data/eval/webshop_415.json

## Candidate-weight training data

Candidate-weight training should use a separate WebShop subset that does
not overlap with WebShop-small.

The intended training subset contains:

    1000 total products
    415 instruction-bearing target products
    585 distractor products
    0 ASIN overlap with WebShop-small
    0 target overlap with the held-out 415 evaluation tasks

The raw full-WebShop files are not stored in this clean repository.

Scripts under scripts/ will be used to:

    download full WebShop data
    construct the non-small 1000-product subset
    construct its 415 instructions
    validate data separation
    build the corresponding search index

## Experiment outputs

Results and logs are intentionally excluded from the clean release.

The following directories should remain untracked:

    results/
    logs/
    archive/
    repeated_runs/
    backup*/
    data/raw_webshop/

---


## Environment setup

The packaged runtime is designed to use relative repository paths.

Recommended runtime:

- Python 3.8
- Java 11 for Pyserini/Lucene
- the Python dependencies in `requirements.txt`
- spaCy model `en_core_web_sm`

From the repository root:

    scripts/setup_environment.sh

Alternatively, install the dependencies manually:

    python -m pip install -r requirements.txt
    python -m spacy download en_core_web_sm

The model endpoint is configured separately through the environment
variables shown below.

## Generic experiment runner

The portable experiment entry point is:

    python -m evaluation.run_experiment --help

Do not invoke the runner as `python evaluation/run_experiment.py`;
module mode is used so repository-local imports resolve correctly.

## Portable execution

All runtime scripts resolve paths relative to the repository root.
No `/workspace/...` path is required after packaging.

### Model endpoint

Configure an OpenAI-compatible endpoint, for example:

    export DEEPSEEK_API_KEY=local-vllm
    export DEEPSEEK_API_BASE=http://127.0.0.1:8000/v1

### WebShop Small evaluation

    scripts/run_small_eval.sh

Optional gate configuration:

    QUERY_GATE=off \
    CANDIDATE_GATE=on \
    PRODUCT_GATE=off \
    MAX_STEPS=20 \
    scripts/run_small_eval.sh results/my_eval

### Candidate weight training

The training set is independent from WebShop Small:

- 1000 products total
- 415 instruction-bearing target products
- 585 distractors
- zero product overlap with WebShop Small
- zero target overlap with the 415 evaluation tasks

Run:

    scripts/train_candidate_weights.sh

Default Candidate weight learning:

- initial type weight: 0.5
- initial requirement weight: 0.5
- eta: 0.1

Override if needed:

    CANDIDATE_INIT_TYPE=0.5 \
    CANDIDATE_INIT_REQ=0.5 \
    CANDIDATE_ONLINE_ETA=0.1 \
    MAX_STEPS=20 \
    scripts/train_candidate_weights.sh

### Final Candidate Selector policy

The packaged Candidate Selector is the final no-threshold version:

1. Candidate Selector is activated only on a repeated product visit.
2. Only visible, unvisited product candidates are considered.
3. A hard contradiction remains a veto.
4. Eligible candidates are ranked by the weighted type/requirement score.
5. The highest-ranked eligible candidate is selected.
6. There is no numeric minimum score threshold.

Historical threshold-based Candidate Selector implementations are not
included in this release.
