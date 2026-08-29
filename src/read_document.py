from pathlib import Path

file_path = Path("data/documents/leave_policy.txt")

content = file_path.read_text(encoding="utf-8")

print(content)