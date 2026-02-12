"""
Check Index Status - Diagnostic Script
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import INDEXES_DIR
from pipeline.database import get_connection
import json

print("="*60)
print("  INDEX DIAGNOSTIC")
print("="*60)

# Check vector store
print("\n[1] Vector Store:")
vectors_path = INDEXES_DIR / "chroma" / "vectors.npy"
metadata_path = INDEXES_DIR / "chroma" / "metadata.json"

if vectors_path.exists():
    import numpy as np
    vectors = np.load(str(vectors_path))
    print(f"    ✓ Vectors file exists: {vectors.shape}")
else:
    print(f"    ✗ No vectors file at {vectors_path}")

if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    print(f"    ✓ Metadata file exists: {len(metadata)} entries")
    if metadata:
        print(f"    Sample entry keys: {list(metadata[0].keys())}")
else:
    print(f"    ✗ No metadata file at {metadata_path}")

# Check keyword index
print("\n[2] Keyword Index:")
keyword_path = INDEXES_DIR / "chroma" / "keyword_index.json"
if keyword_path.exists():
    kw_data = json.loads(keyword_path.read_text())
    print(f"    ✓ Keyword index exists")
    print(f"    Terms indexed: {len(kw_data.get('inverted_index', {}))}")
    print(f"    Chunks indexed: {len(kw_data.get('chunk_texts', {}))}")
else:
    print(f"    ✗ No keyword index at {keyword_path}")

# Check database
print("\n[3] Database:")
try:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM artifacts")
    artifacts = cursor.fetchone()[0]
    print(f"    Artifacts: {artifacts}")

    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunks = cursor.fetchone()[0]
    print(f"    Chunks: {chunks}")

    cursor.execute("SELECT COUNT(*) FROM evaluations")
    evals = cursor.fetchone()[0]
    print(f"    Evaluations: {evals}")

    if artifacts > 0:
        cursor.execute("SELECT source_kind, COUNT(*) FROM artifacts GROUP BY source_kind")
        print(f"\n    Artifacts by source_kind:")
        for row in cursor.fetchall():
            print(f"      {row[0]}: {row[1]}")

    conn.close()
except Exception as e:
    print(f"    ✗ Database error: {e}")

# Check if runs directory has eval report
print("\n[4] Runs Directory:")
runs_dir = Path(__file__).parent.parent / "runs"
if runs_dir.exists():
    weeks = list(runs_dir.iterdir())
    print(f"    Weeks found: {[w.name for w in weeks if w.is_dir()]}")
    for week_dir in weeks:
        if week_dir.is_dir():
            eval_report = week_dir / "eval_report.json"
            if eval_report.exists():
                print(f"    ✓ {week_dir.name}/eval_report.json exists")
else:
    print(f"    ✗ No runs directory")

print("\n" + "="*60)
print("  RECOMMENDATION")
print("="*60)

if not vectors_path.exists() or not metadata_path.exists():
    print("\n  ⚠ Index is EMPTY. Run: python scripts/inject_demo_data.py")
else:
    print("\n  ✓ Index has data. If queries still fail, check retriever loading.")
