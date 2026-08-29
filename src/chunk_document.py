from pathlib import Path

file_path = Path("data/documents/leave_policy.txt")

content = file_path.read_text(encoding="utf-8")

chunks = content.split("\n\n")

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)