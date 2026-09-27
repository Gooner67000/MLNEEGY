# Deploying the platform

There are three ways to run it. The first two are the full platform; the third is
the single-model demo.

| | Who runs it | Where data lives | Cost | Good for |
|---|---|---|---|---|
| **A. Download & run** | The customer, on their own PC/server | Their machine only | Free | Factories that won't send sensor data off-site; pilots |
| **B. Hosted (Render)** | You | Render's cloud (Postgres) | Free to try; paid for real customers | Demos, SaaS customers |
| **C. Single-model demo** | You | — | Free | Portfolio / interview link |

---

## A. Download & run (on-prem)

Needs [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or
Docker Engine (Linux).

```bash
git clone https://github.com/Gooner67000/MLNEEGY.git
cd MLNEEGY
docker compose up --build
```

- Web app: http://localhost:8501
- API + interactive docs: http://localhost:8000/docs

Nothing leaves the machine. Data is kept in `./data/app.db` between restarts. Every push
to this repo builds this exact stack in CI and scores a machine of all 7 types through it,
so it's known to work.

---

## B. Hosted on Render (one click + one paste)

`render.yaml` in this repo defines everything: the API, the web app, and a Postgres
database. You need a Render account. Sign in with GitHub. It's free to start, and
**only you can create it** — it's your account and your billing.

1. Click **[Deploy to Render](https://render.com/deploy?repo=https://github.com/Gooner67000/MLNEEGY)**
   and sign in with GitHub.
2. Render reads `render.yaml` and shows three resources: `pm-backend`, `pm-frontend`, `pm-db`.
3. It asks for two values, `API_BASE` and `PUBLIC_API_BASE`. You don't know the backend's
   URL yet, so enter `https://pm-backend.onrender.com` for both, then click **Apply**.
4. When `pm-backend` finishes deploying, copy its real URL from its Render page. It might
   carry a suffix, e.g. `https://pm-backend-ab12.onrender.com`. If it differs from what you
   typed, open **pm-frontend → Environment**, paste it into both `API_BASE` and
   `PUBLIC_API_BASE`, and save. The web app redeploys by itself.
5. Open the `pm-frontend` URL and register your business.

What's already secured:
- `SECRET_KEY` is generated randomly by Render. The backend also runs with
  `PM_ENV=production`, which **refuses to start** if the secret is weak or left at its
  default, so login tokens can't be forged.
- Passwords are bcrypt-hashed. Each business only sees its own machines (tested in CI).

**Free-tier limits** (from [render.com/docs/free](https://render.com/docs/free), Sept 2026):

| Limit | What it means for you |
|---|---|
| Free web services sleep after **15 min** idle; about a **1 minute** cold start | Fine for demos. For live customer data, the first reading after a quiet period is slow. |
| Free Postgres **expires 30 days** after creation (then 14-day grace, then deleted) | **Don't keep real customer data on the free database.** Upgrade `pm-db` to a paid plan before onboarding anyone. |
| **750** free instance-hours per month per workspace | Two always-on services use about 1,440 h/month. Sleeping keeps you under, but heavy use can suspend them until the month resets. |
| No persistent disk on free web services | Why the hosted version uses Postgres, not SQLite |

Before treating it as a paid product, you'll also need a privacy policy and terms of
service, a data-processing agreement for customers, and backups. Those are business and
legal decisions, not code.

---

## C. Single-model demo (Streamlit Community Cloud)

The original CNC-only `app.py` can be a free public link:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → repo `Gooner67000/MLNEEGY`, branch `main`, main file `app.py` → **Deploy**.
