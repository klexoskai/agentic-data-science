import sys
sys.path.insert(0, ".")
from store.client import get_client, get_collection
from store.config import ALL_COLLECTIONS

client = get_client()
print("=== Collections ===")
for name in ALL_COLLECTIONS:
    try:
        col = get_collection(name)
        count = col.count()
        print(f"  {name}: {count} chunks")
        if count > 0:
            sample = col.peek(limit=2)
            for doc, meta in zip(sample["documents"], sample["metadatas"]):
                preview = doc[:120]
                source = meta.get("source", "?")
                print(f"    [{source}] {preview}")
    except Exception as e:
        print(f"  {name}: error - {e}")