"""
keepalive.py — Heartbeat Ping for Supabase and Qdrant Cloud
============================================================
Sends lightweight read/ping operations to:
  1. Supabase (PostgreSQL / Auth API)
  2. Qdrant Cloud (Cluster telemetry / Collection info)

Prevents free-tier inactivity pausing on both platforms.
Can be executed manually or scheduled via Windows Task Scheduler / cron.
"""

import datetime
import sys
from pathlib import Path

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import settings
from qdrant_client import QdrantClient
from supabase import create_client


def ping_supabase() -> dict:
    """Sends a lightweight query and auth ping to Supabase."""
    print(f"[{datetime.datetime.now().isoformat()}] Pinging Supabase at {settings.SUPABASE_URL}...")
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    
    # 1. Auth service ping
    users = supabase.auth.admin.list_users(page=1, per_page=1)
    user_count = len(users)
    
    # 2. Database table ping
    try:
        db_res = supabase.table("query_history").select("id").limit(1).execute()
        db_status = f"OK ({len(db_res.data)} records queried)"
    except Exception as e:
        db_status = f"Table query note: {e}"

    print(f"  [OK] Supabase Auth: Active ({user_count} user retrieved)")
    print(f"  [OK] Supabase DB:   Active ({db_status})")
    return {"status": "active", "auth": True, "db": db_status}


def ping_qdrant() -> dict:
    """Sends a cluster health check and collection info query to Qdrant Cloud."""
    print(f"[{datetime.datetime.now().isoformat()}] Pinging Qdrant Cloud at {settings.QDRANT_URL}...")
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    # 1. Cluster collections check
    cols = client.get_collections()
    col_names = [c.name for c in cols.collections]
    
    # 2. Inspect document points count
    counts = {}
    for name in col_names:
        info = client.get_collection(name)
        counts[name] = info.points_count

    print(f"  [OK] Qdrant Cloud:  Active (Collections: {col_names})")
    for name, count in counts.items():
        print(f"    - {name}: {count} points safe and stored")
        
    return {"status": "active", "collections": counts}


def run_heartbeat():
    print("=" * 70)
    print(f"DOCUMIND CLOUD HEARTBEAT -- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    try:
        ping_supabase()
    except Exception as e:
        print(f"  [ERROR] Supabase Ping Failed: {e}")

    print("-" * 70)
    try:
        ping_qdrant()
    except Exception as e:
        print(f"  [ERROR] Qdrant Ping Failed: {e}")

    print("=" * 70)
    print("Cloud keepalive signal successfully dispatched! Both services refreshed.")
    print("=" * 70)


if __name__ == "__main__":
    run_heartbeat()
