import sys
from pathlib import Path

# make `import summarization` work when running `pytest` from connect_interview_ai/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
