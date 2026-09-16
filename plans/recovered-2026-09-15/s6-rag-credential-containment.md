# S6 — RAG credential / transport containment runbook

**Do immediately** (do not wait on crawl SSRF polish).

## Blocker before loopback bind

Live Hostinger compose still publishes `0.0.0.0:8080`. Repo `deploy/hostinger-compose.yml` already uses `127.0.0.1:8080:8080`.

**Do not apply the loopback bind alone** while GeekAPI still calls `http://2.24.101.90:8080` — that breaks production RAG. Order:

1. Stand up TLS terminator (Caddy/nginx) on the VPS listening publicly on 443 → `127.0.0.1:8080`
2. Point GeekAPI `GEEK_CRAWLER_RAG_URL` at the HTTPS hostname
3. Then apply compose bind to loopback + restart
4. Rotate `API_KEY` / Railway `GEEK_CRAWLER_RAG_API_KEY` in the same window

Hostinger MCP `VPS_updateProject` only recreates from the on-box compose file; it does **not** push repo compose edits. Edit `/docker/geek-crawler-rag/docker-compose.yml` over SSH (or panel), then `docker compose up -d`.

## Related exposure

The colocated `mongodb` project currently publishes `0.0.0.0:27017`. Bind Mongo to `127.0.0.1` (or firewall drop) in the same containment window — not required for RAG TLS, but it is the same class of plaintext public listener.

## 1. Rotate the API key

```bash
# Generate a new key
openssl rand -hex 32

# On Hostinger VPS: update geek-crawler-rag .env API_KEY=...
# On Railway GeekAPI: set GEEK_CRAWLER_RAG_API_KEY to the same value
# Redeploy / restart both
```

Confirm old key returns 401:

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "X-Api-Key: OLD_KEY" http://127.0.0.1:8080/v1/capabilities
# expect 401
```

## 2. Cut public plaintext HTTP

`deploy/hostinger-compose.yml` now binds `127.0.0.1:8080:8080` only.

```bash
cd /docker/geek-crawler-rag   # or your compose path
# After TLS is live and GeekAPI points at https://...
docker compose up -d api
```

Verify from outside the VPS that `:8080` is no longer reachable; on-box loopback still serves health for TLS terminator / SSH tunnel.

## 3. Put TLS + hostname in front

DNS: `rag.geekatyourspot.com` → VPS IP. Caddy assets: [`deploy/Caddyfile`](../deploy/Caddyfile) + [`deploy/caddy-compose.yml`](../deploy/caddy-compose.yml) — see [`deploy/README-caddy.md`](../deploy/README-caddy.md).

Point GeekAPI `GEEK_CRAWLER_RAG_URL` at `https://rag.geekatyourspot.com` and **clear** `GEEK_CRAWLER_RAG_ALLOW_INSECURE_HTTP_HOSTS`. Localhost HTTP remains allowed only for local GeekAPI→RAG.

**This machine has no SSH key to the VPS** (`Permission denied (publickey)`). Caddy install + compose bind + key rotate must be done from a shell that can reach `root@2.24.101.90` (or Hostinger panel terminal).

## 4. Production signal

- Zero successful RAG calls with the retired key
- GeekAPI logs show `https://` RAG base address
- Public probe of former bare-IP `:8080` fails
