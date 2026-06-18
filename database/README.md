# Tracking Database — Access Guide

PostgreSQL 18 running in Docker, database `tracking`, user `sipg`.

---

## 1. Inside the container (psql)

The quickest way to run queries directly.

```bash
docker exec -it tracking-db psql -U sipg -d tracking
```

Useful psql commands once connected:

```sql
\dt                  -- list all tables
\d fact_detection    -- describe a table
\q                   -- quit
```

---

## 2. From your Mac (pipe SQL file)

Run a `.sql` file from your Mac's filesystem without entering the container.

```bash
docker exec -i tracking-db psql -U sipg -d tracking < /path/to/file.sql
```

---

## 3. From your Mac (psql client)

Install the Postgres client via Homebrew, then connect directly over the mapped port.

```bash
# Install once
brew install libpq
echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Connect
psql -h localhost -p 5432 -U sipg -d tracking
```

Password: `sipg`

---

## 4. SSH tunnel (remote access)

If the database is running on a remote server, tunnel through SSH so you never
expose port 5432 to the internet.

```bash
# On your local machine — keep this terminal open
ssh -L 5432:localhost:5432 user@your-server-ip

# In a second terminal
psql -h localhost -p 5432 -U sipg -d tracking
```

---

## 5. GUI client (TablePlus / DBeaver / pgAdmin)

Use any Postgres-compatible GUI with these connection details:

| Field    | Value       |
|----------|-------------|
| Host     | `localhost` |
| Port     | `5432`      |
| Database | `tracking`  |
| Username | `sipg`      |
| Password | `sipg`      |

For a remote server, set up the SSH tunnel (method 4) first, then connect to `localhost:5432`.

---

## 6. From Python (psycopg2)

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="tracking",
    user="sipg",
    password="sipg"
)

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM fact_detection;")
print(cur.fetchone())
cur.close()
conn.close()
```

Install the driver with:

```bash
pip install psycopg2-binary
```

---

## 7. From Python (SQLAlchemy)

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://sipg:sipg@localhost:5432/tracking")

with engine.connect() as conn:
    result = conn.execute("SELECT COUNT(*) FROM fact_detection;")
    print(result.fetchone())
```

Install with:

```bash
pip install sqlalchemy psycopg2-binary
```

---

## Container management

```bash
# Start / stop (data is preserved)
docker start tracking-db
docker stop tracking-db

# Check container is running
docker ps

# View logs
docker logs tracking-db

# Confirm volume is mounted
docker inspect tracking-db | grep -A 10 Mounts
```

---

## Data persistence

Data is stored in a Docker named volume:

| Field       | Value                                        |
|-------------|----------------------------------------------|
| Volume name | `postgres-data`                              |
| Mount path  | `/var/lib/postgresql` (inside container)     |
| Host path   | `/var/lib/docker/volumes/postgres-data/_data`|

Data survives container restarts. To permanently delete it:

```bash
# ⚠️ This deletes all data
docker rm -f tracking-db
docker volume rm postgres-data
```


CSV columns expected:
    frame_id, timestamp_s, track_id, class_id, class_name, state,
    x1, y1, x2, y2, width, height, cx, cy

Use ingest.py to add things to database

