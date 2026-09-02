import json
from pathlib import Path

from app.bootstrap import build_services
from app.evaluation import evaluate
from app.ingest import load_sample_documents
from app.models import EvalCase


def load_cases() -> list[EvalCase]:
    path = Path(__file__).resolve().parent.parent / "sample_data" / "eval.json"
    data = json.loads(path.read_text())
    return [EvalCase(**item) for item in data]


def main() -> None:
    services = build_services()
    services.ingestion.ingest(load_sample_documents())
    summary = evaluate(services.agent, load_cases())
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
