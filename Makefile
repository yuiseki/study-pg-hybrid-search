
all: start

.PHONY: build
build:
	docker compose build

.PHONY: start
start:
	docker compose up -d --build

.PHONY: logs
logs:
	docker compose logs -f

.PHONY: stop
stop:
	docker compose down

.PHONY: clean
clean:
	docker compose down -v --remove-orphans
	docker run --rm -v $$(pwd)/db/data:/work busybox sh -c 'rm -rf /work/* /work/.[!.]* /work/..?*'
	mkdir -p ./db/data

.PHONY: psql
psql:
	docker compose exec db psql -U postgres -d sandbox
