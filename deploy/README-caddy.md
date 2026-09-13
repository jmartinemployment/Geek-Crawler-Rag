# Caddy TLS front for Geek-Crawler-Rag (S6)

Proxies `https://rag.geekatyourspot.com` → `http://127.0.0.1:8080`.

## Prerequisites

1. DNS: `rag.geekatyourspot.com` A → VPS public IP (`2.24.101.90`)
2. Firewall: allow TCP 80 + 443 (keep 22); remove public 8080 after cutover
3. RAG compose binds `127.0.0.1:8080:8080` (see `hostinger-compose.yml`)
4. On the VPS, place this file next to a compose override or run as its own project

## Install on VPS (SSH)

```bash
mkdir -p /docker/rag-caddy
cp deploy/Caddyfile /docker/rag-caddy/Caddyfile
cp deploy/caddy-compose.yml /docker/rag-caddy/docker-compose.yml
cd /docker/rag-caddy
docker compose up -d
```

## Cutover order

1. Start Caddy (this compose) while RAG is still public on `:8080` if needed for dual-run validation
2. `curl -fsS https://rag.geekatyourspot.com/v1/capabilities` → expect 401 without key
3. Railway GeekAPI: set `GEEK_CRAWLER_RAG_URL=https://rag.geekatyourspot.com` and clear `GEEK_CRAWLER_RAG_ALLOW_INSECURE_HTTP_HOSTS`
4. Edit `/docker/geek-crawler-rag/docker-compose.yml` ports to `127.0.0.1:8080:8080` and `docker compose up -d`
5. Rotate `API_KEY` / Railway `GEEK_CRAWLER_RAG_API_KEY`
6. Firewall: drop public 8080 accept rule
