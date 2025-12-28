# Default target
all: start

COMPOSE ?= docker compose
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
MODELS ?= snowflake-arctic-embed:22m nomic-embed-text:v1.5 embeddinggemma:300m snowflake-arctic-embed2:568m qwen3-embedding:4b
EVAL_ARGS ?=

.PHONY: build
build:
	$(COMPOSE) build

.PHONY: start
start:
	$(COMPOSE) up -d --build

.PHONY: logs
logs:
	$(COMPOSE) logs -f

.PHONY: stop
stop:
	$(COMPOSE) down

.PHONY: clean
clean:
	$(COMPOSE) down -v --remove-orphans
	docker run --rm -v $$(pwd)/db/data:/work busybox sh -c 'rm -rf /work/* /work/.[!.]* /work/..?*'
	mkdir -p ./db/data

.PHONY: psql
psql:
	$(COMPOSE) exec db psql -U postgres -d sandbox

.PHONY: reset
reset: clean start

.PHONY: venv
venv:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: embed_all
embed_all: venv
	@for model in $(MODELS); do \
		$(PYTHON) scripts/embed_documents.py --model $$model ; \
	 done

.PHONY: eval
eval: venv
	$(PYTHON) evaluations/run_eval.py --modes text,vector,hybrid --rrf-weights 1:1 2:1 1:2 $(EVAL_ARGS)
	$(PYTHON) evaluations/calc_metrics.py --rankings $$(ls -t evaluations/out/*/rankings.jsonl | head -n1)

.PHONY: eval_models
eval_models: venv
	$(PYTHON) evaluations/run_eval.py --models "$(MODELS)" $(EVAL_ARGS)
