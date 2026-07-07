#!/usr/bin/env python3
"""Validate the python-development skill tree with stdlib-only checks."""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references"
EXAMPLES = ROOT / "examples"
QUALITY = ROOT / "quality"

REQUIRED_SCHEMA_FIELDS = {
    "file",
    "line",
    "symbol",
    "category",
    "severity",
    "title",
    "why",
    "fix",
    "validation",
}

CATEGORIES = {
    "correctness",
    "api",
    "design",
    "typing",
    "errors",
    "testing",
    "io",
    "performance",
    "concurrency",
    "security",
    "packaging",
    "maintainability",
    "style",
    "tooling",
}

SEVERITIES = {"high", "medium", "low"}
MODES = {
    "planning",
    "feature",
    "bugfix",
    "refactor",
    "testing",
    "review",
    "debugging",
    "security",
    "typing",
    "io",
    "concurrency",
    "packaging",
}

FORBIDDEN_TERMS = [
    "mini" + "COIL",
    "mini" + "coil",
    "her" + "ald",
    "kv" + "press",
    "GSM" + "8K",
    "ori" + "on",
    "src/" + "her" + "ald",
]

FORBIDDEN_CHARS = {
    "em dash": chr(0x2014),
}

DIMENSIONS = {
    "discovery",
    "contract",
    "test_strategy",
    "implementation_simplicity",
    "validation_evidence",
    "review_discipline",
    "safety",
    "project_adaptation",
    "reporting_clarity",
}


def main() -> int:
    """Run every validation check and print a compact report."""
    failures: list[str] = []
    checks = 0

    for check in CHECKS:
        checks += 1
        try:
            failures.extend(check())
        except FileNotFoundError as exc:
            failures.append(f"missing file required by validation check: {exc.filename}")

    if failures:
        print("python-development skill validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    ref_count = len(list(REFERENCES.glob("*.md")))
    print(
        "python-development skill validation passed: "
        f"{checks} check groups, {ref_count} reference files"
    )
    return 0


def check_required_paths() -> list[str]:
    """Check that required top-level files and directories exist."""
    required = [
        ROOT / "SKILL.md",
        REFERENCES / "README.md",
        REFERENCES / "review-schema.md",
        EXAMPLES / "review-findings.json",
        EXAMPLES / "validation-reports.md",
        QUALITY / "README.md",
        QUALITY / "sample_tasks.json",
        QUALITY / "scorecard.md",
        QUALITY / "scorecard.schema.json",
        QUALITY / "fixtures" / "README.md",
    ]
    return [f"missing required path: {path}" for path in required if not path.exists()]


def check_front_matter() -> list[str]:
    """Check basic skill metadata in the YAML front matter block."""
    text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    failures: list[str] = []
    if not text.startswith("---\n"):
        return ["SKILL.md must start with YAML front matter"]

    parts = text.split("---", 2)
    if len(parts) < 3:
        return ["SKILL.md front matter must have an opening and closing fence"]

    front_matter = parts[1]
    fields: dict[str, str] = {}
    for raw_line in front_matter.splitlines():
        if not raw_line.strip() or raw_line.startswith(" "):
            continue
        if ":" not in raw_line:
            failures.append(f"invalid front matter line: {raw_line}")
            continue
        key, value = raw_line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"')

    if fields.get("name") != "python-development":
        failures.append("SKILL.md front matter must name python-development")
    if not fields.get("description"):
        failures.append("SKILL.md front matter must include a description")
    return failures


def check_reference_index() -> list[str]:
    """Check that the reference index and reference files agree."""
    readme = (REFERENCES / "README.md").read_text(encoding="utf-8")
    reference_files = sorted(
        path.name for path in REFERENCES.glob("*.md") if path.name != "README.md"
    )
    failures: list[str] = []

    for name in reference_files:
        if f"`{name}`" not in readme:
            failures.append(f"reference file not indexed in references/README.md: {name}")

    for name in re.findall(r"`([^`]+\.md)`", readme):
        if name == "SKILL.md":
            continue
        if not (REFERENCES / name).exists():
            failures.append(f"references/README.md lists missing file: {name}")

    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    for name in reference_files:
        if f"references/{name}" not in skill and name not in {"README.md"}:
            if name in {"maintenance.md"}:
                continue
            failures.append(f"SKILL.md reference catalogue omits: {name}")
    return failures


def check_sources_sections() -> list[str]:
    """Check that reference files carry source sections."""
    failures: list[str] = []
    for path in REFERENCES.glob("*.md"):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        if "## Sources" not in text:
            failures.append(f"reference file missing '## Sources': {path.name}")
    return failures


def check_forbidden_content() -> list[str]:
    """Check forbidden leakage terms, placeholders, and punctuation."""
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if should_skip_path(path):
            continue
        checked_suffixes = {".md", ".json", ".py", ".toml", ".yml", ".yaml", ".txt"}
        if not path.is_file() or path.suffix not in checked_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            if term.lower() in text.lower():
                failures.append(f"forbidden project-specific term in {relative(path)}: {term}")
        for name, char in FORBIDDEN_CHARS.items():
            if char in text:
                failures.append(f"forbidden punctuation in {relative(path)}: {name}")
        stale_marker = "PLACE" + "HOLDER"
        if stale_marker in text:
            failures.append(f"stale placeholder text in {relative(path)}")
    return failures


def check_review_examples() -> list[str]:
    """Check review finding examples against the documented schema."""
    path = EXAMPLES / "review-findings.json"
    data = load_json(path)
    failures: list[str] = []
    if not isinstance(data, dict):
        return ["review-findings.json must contain an object"]

    findings = collect_findings(data)
    if not findings:
        failures.append("review-findings.json must include at least one finding example")
    for index, finding in enumerate(findings):
        missing = REQUIRED_SCHEMA_FIELDS - finding.keys()
        if missing:
            failures.append(
                f"review finding {index} missing fields: {', '.join(sorted(missing))}"
            )
        extra = set(finding) - REQUIRED_SCHEMA_FIELDS
        if extra:
            failures.append(
                f"review finding {index} has extra fields: {', '.join(sorted(extra))}"
            )
        category = finding.get("category")
        if category not in CATEGORIES:
            failures.append(f"review finding {index} has unknown category: {category}")
        severity = finding.get("severity")
        if severity not in SEVERITIES:
            failures.append(f"review finding {index} has unknown severity: {severity}")
    return failures


def check_sample_tasks() -> list[str]:
    """Check sample task records used by the manual harness."""
    path = QUALITY / "sample_tasks.json"
    data = load_json(path)
    failures: list[str] = []
    if not isinstance(data, list):
        return ["sample_tasks.json must contain a list"]

    fixture_names = documented_fixture_names()
    documented_tasks = (QUALITY / "sample-tasks.md").read_text(encoding="utf-8")
    required = {
        "id",
        "mode",
        "prompt",
        "fixture",
        "required_behaviors",
        "forbidden_behaviors",
        "primary_references",
        "scoring_focus",
    }
    ids: set[str] = set()
    for index, task in enumerate(data):
        if not isinstance(task, dict):
            failures.append(f"sample task {index} must be an object")
            continue
        missing = required - task.keys()
        if missing:
            failures.append(
                f"sample task {index} missing fields: {', '.join(sorted(missing))}"
            )
        task_id = str(task.get("id", ""))
        if task_id in ids:
            failures.append(f"duplicate sample task id: {task_id}")
        ids.add(task_id)
        if task_id and f"`{task_id}`" not in documented_tasks:
            failures.append(f"sample task {task_id} is not documented in sample-tasks.md")
        if task.get("mode") not in MODES:
            failures.append(f"sample task {task_id} has invalid mode: {task.get('mode')}")
        fixture = task.get("fixture")
        if fixture not in fixture_names:
            failures.append(f"sample task {task_id} references undocumented fixture: {fixture}")
        for field in (
            "required_behaviors",
            "forbidden_behaviors",
            "primary_references",
            "scoring_focus",
        ):
            value = task.get(field)
            if not is_non_empty_string_list(value):
                failures.append(f"sample task {task_id} needs non-empty string list {field}")
        for ref in task.get("primary_references", []):
            if not (REFERENCES / ref).exists():
                failures.append(f"sample task {task_id} references missing file: {ref}")
        for dimension in task.get("scoring_focus", []):
            if dimension not in DIMENSIONS:
                failures.append(
                    f"sample task {task_id} uses unknown scorecard dimension: {dimension}"
                )
    return failures


def check_scorecard_schema() -> list[str]:
    """Check scorecard JSON schema and documented dimensions."""
    schema = load_json(QUALITY / "scorecard.schema.json")
    example = load_json(QUALITY / "scorecard.example.json")
    sample_tasks = load_json(QUALITY / "sample_tasks.json")
    sample_task_ids = {
        task.get("id") for task in sample_tasks if isinstance(task, dict)
    }
    failures: list[str] = []
    text = (QUALITY / "scorecard.md").read_text(encoding="utf-8")
    for dimension in DIMENSIONS:
        if dimension not in text:
            failures.append(f"scorecard.md omits dimension: {dimension}")
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    score_properties = properties.get("scores", {}).get("properties", {})
    schema_dimensions = set(score_properties)
    if schema_dimensions != DIMENSIONS:
        failures.append(
            "scorecard.schema.json score dimensions differ from expected set: "
            f"{sorted(schema_dimensions)}"
        )
    failures.extend(validate_scorecard_example(example, sample_task_ids))
    return failures


def check_all_json_files_parse() -> list[str]:
    """Check every JSON file in the skill tree parses."""
    failures: list[str] = []
    for path in ROOT.rglob("*.json"):
        if should_skip_path(path):
            continue
        try:
            load_json(path)
        except SystemExit as exc:
            failures.append(str(exc))
    return failures


def check_schema_documentation_alignment() -> list[str]:
    """Check that documented review schemas use the canonical fields."""
    failures: list[str] = []
    for relative_path in ("SKILL.md", "references/review-schema.md"):
        path = ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        for field in REQUIRED_SCHEMA_FIELDS:
            if f'"{field}"' not in text:
                failures.append(f"{relative_path} review schema omits field: {field}")
        for category in CATEGORIES:
            if category not in text:
                failures.append(f"{relative_path} review schema omits category: {category}")
        for severity in SEVERITIES:
            if severity not in text:
                failures.append(f"{relative_path} review schema omits severity: {severity}")
    return failures


def check_python_files() -> list[str]:
    """Check Python files for syntax and function docstrings."""
    failures: list[str] = []
    for path in ROOT.rglob("*.py"):
        if should_skip_path(path):
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            failures.append(f"Python syntax error in {relative(path)}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if ast.get_docstring(node) is None:
                    failures.append(
                        f"Python function missing docstring in {relative(path)}: {node.name}"
                    )
    return failures


def validate_scorecard_example(example: Any, sample_task_ids: set[Any]) -> list[str]:
    """Validate the bundled scorecard example against key schema expectations."""
    failures: list[str] = []
    if not isinstance(example, dict):
        return ["scorecard.example.json must contain an object"]
    required = {
        "task_id",
        "reviewer",
        "date",
        "scores",
        "notes",
        "command_evidence",
        "residual_risks",
        "final_verdict",
    }
    missing = required - example.keys()
    if missing:
        failures.append(
            "scorecard.example.json missing fields: " + ", ".join(sorted(missing))
        )
    scores = example.get("scores", {})
    if not isinstance(scores, dict):
        failures.append("scorecard.example.json scores must be an object")
        return failures
    if set(scores) != DIMENSIONS:
        failures.append(
            "scorecard.example.json score dimensions differ from expected set: "
            f"{sorted(scores)}"
        )
    for dimension, score in scores.items():
        if not isinstance(score, int) or score < 0 or score > 2:
            failures.append(f"scorecard.example.json invalid score for {dimension}: {score}")
    if example.get("task_id") not in sample_task_ids:
        failures.append("scorecard.example.json task_id does not match a sample task")
    if example.get("final_verdict") not in {"pass", "marginal", "fail"}:
        failures.append("scorecard.example.json final_verdict is invalid")
    return failures


def load_json(path: Path) -> Any:
    """Load JSON from a path and raise a readable validation failure on errors."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def collect_findings(data: Any) -> list[dict[str, Any]]:
    """Collect finding-like objects from nested example payloads."""
    findings: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if is_finding_like(data):
            findings.append(data)
        for key, value in data.items():
            if key == "findings" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        findings.append(item)
                    else:
                        findings.extend(collect_findings(item))
            else:
                findings.extend(collect_findings(value))
    elif isinstance(data, list):
        for item in data:
            findings.extend(collect_findings(item))
    return findings


def is_finding_like(value: dict[str, Any]) -> bool:
    """Return true when an object resembles a review finding."""
    return bool(REQUIRED_SCHEMA_FIELDS & value.keys()) and any(
        key in value for key in ("title", "why", "fix", "severity", "category")
    )


def documented_fixture_names() -> set[str]:
    """Return fixture names documented in quality/fixtures/README.md."""
    text = (QUALITY / "fixtures" / "README.md").read_text(encoding="utf-8")
    names: set[str] = set()
    for line in text.splitlines():
        if line.startswith("## "):
            names.add(line.removeprefix("## ").strip())
    return names


def is_non_empty_string_list(value: Any) -> bool:
    """Return true when a value is a non-empty list of non-empty strings."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def should_skip_path(path: Path) -> bool:
    """Return true for generated or environment paths the validator should ignore."""
    skip_parts = {".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return bool(skip_parts & set(path.parts))


def relative(path: Path) -> str:
    """Return a path relative to the skill root when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


CHECKS = [
    check_required_paths,
    check_front_matter,
    check_reference_index,
    check_sources_sections,
    check_forbidden_content,
    check_review_examples,
    check_sample_tasks,
    check_scorecard_schema,
    check_all_json_files_parse,
    check_schema_documentation_alignment,
    check_python_files,
]

if __name__ == "__main__":
    sys.exit(main())
