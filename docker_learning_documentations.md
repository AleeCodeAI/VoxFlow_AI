# Docker Learnings — 3 Days of Pain 😭

## Docker Fundamentals

- **Images vs containers vs volumes** — image is the blueprint, container is the running instance, volume is persistent storage
- `docker compose up` — creates and starts containers
- `docker compose down` — stops and removes containers
- `docker compose restart` — restarts containers but doesn't reload env files
- `docker compose build` — builds images from Dockerfile
- `docker logs <container> -f` — follow live logs
- `docker exec -it <container> <command>` — run a command inside a running container
- `docker ps` — list running containers
- `docker kill <container>` — force stop a stuck container
- Code changes require a **rebuild** — `docker compose build`
- Config/env changes only need a **restart** — `docker compose restart`

---

## Networking

- Host port vs container port — `5434:5432` means left side for humans, right side for containers
- **Containers talk to each other using service names** — `shared-postgres`, `shared-redis`, not `localhost`
- `localhost` inside a container means the container itself, not your machine
- Named external networks allow multiple projects to share the same containers
- Browser always uses host port (left), containers always use internal port (right)

```
Your browser   → localhost:3100  (host port)
Other containers → shared-langfuse:3000  (container port)
```

---

## Environment Variables

- **Priority order:** compose `environment` → `env_file` → class defaults
- Pydantic `BaseSettings` only reads variables defined as fields — `extra="ignore"` silently drops everything else
- Secrets belong in `.env`, never in images or compose files
- `.dockerignore` keeps secrets out of images
- Same codebase, different behavior in Docker vs local dev — just by overriding variables

```python
# Pydantic priority
class MainSettings(BaseSettings):
    REDIS_PORT: int = 6379  # default — lowest priority
```

```yaml
# compose environment — highest priority, overrides everything
environment:
  - REDIS_PORT=6379
```

```env
# .env file — middle priority
REDIS_PORT=6380
```

---

## Shared Infrastructure Pattern

- One PostgreSQL — multiple databases per project
- One Langfuse — multiple projects with separate API keys
- One Redis and one n8n — shared by all projects
- Projects only contain their own app containers (backend, frontend)
- Shared infra lives in its own folder `C:\DEV\infrastructure`

```yaml
# Other projects reference the shared network like this
networks:
  infra-network:
    external: true
```

---

## Local Dev vs Docker — The Clean Pattern

Keep local `.env` as is for local development.
Add Docker-specific values in compose `environment` — they override local ones automatically.

```env
# backend/.env — for local dev
POSTGRESQL_URL="postgresql://postgres:password@localhost:5434/mydb"
LANGFUSE_HOST="http://localhost:3100"
N8N_WEBHOOK_URL="http://localhost:5679/webhook/abc123"
```

```yaml
# docker-compose.yml — overrides for Docker
environment:
  - POSTGRESQL_URL=postgresql://postgres:password@shared-postgres:5432/mydb
  - LANGFUSE_HOST=http://shared-langfuse:3000
  - N8N_WEBHOOK_URL=http://shared-n8n:5678/webhook/abc123
```

- Local dev → reads `.env` → uses `localhost` URLs
- Docker → compose overrides → uses container service names
- Same code, zero changes needed when switching contexts

---

## Dockerfile Best Practices

### `uv sync` at build time not runtime
`uv sync` installs all packages. It should run during `docker build` (baked into the image) not when the container starts. Running it at startup means 125 packages install every single time the container starts — very slow.

### `CMD` with direct venv path instead of `uv run`
```dockerfile
# Bad — triggers uv sync check on every container start
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Good — runs directly from venv, no uv involved at runtime
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
`uv run` is a development convenience tool. In production containers run the binary directly from the venv.

### Layer caching — order matters
Docker caches each layer. If a layer hasn't changed, it skips it on rebuild. Correct order = fast rebuilds.

```dockerfile
# Changes rarely — almost always cached
RUN apt-get install ffmpeg...
RUN curl uv install...

# Changes only when you add/remove packages
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Changes constantly — never cached, always runs
COPY . .
```

If you put `COPY . .` before `uv sync`, every code change would invalidate the package install layer and reinstall everything from scratch.

### `UV_LINK_MODE=copy`
uv tries to hardlink files from its cache to save disk space. Docker's filesystem doesn't support hardlinks across layers so it falls back to copying and prints a warning. Setting `UV_LINK_MODE=copy` tells uv to just copy from the start — no warning, same result.

```dockerfile
ENV UV_LINK_MODE=copy
```

### Final clean Dockerfile pattern
```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y \
    ffmpeg curl gcc g++ libcairo2-dev pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Debugging Inside Containers

```bash
# Check what environment variables the container actually has
docker exec -it <container> env | findstr <VARIABLE>

# Test what pydantic actually loads
docker exec -it <container> python -c "from configs.main_configs import MainSettings; s = MainSettings(); print(s.VARIABLE)"

# Check if a file exists inside the container
docker exec -it <container> cat /app/.env

# Run a module inside the container
docker exec -it <container> python -m module.name
```

---

## Lessons Learned The Hard Way 😂

- **Always check your git branch before rebuilding** — wrong branch = wrong code in image
- **Verify all changed files before starting a build** — the build is expensive
- `restart` doesn't reload env files — use `down` + `up`
- A variable in compose `environment` does nothing if it's not a field in `MainSettings`
- `localhost` inside Docker never means your machine
- Layer order in Dockerfile directly affects rebuild speed
- Secrets never go in images — `.dockerignore` your `.env`
- Host ports are for you, container ports are for containers