"""
Daily AI commit agent.

Picks one repo + one task type for today, asks Groq's API to make a small,
real improvement to a file, validates it compiles, and commits + pushes.
Also logs a short entry to this control repo (ai-learning-log).

Modes:
  DRY_RUN=true   (default) -> generates and PRINTS the change only.
                                Nothing is written, committed, or pushed.
  DRY_RUN=false             -> applies, validates, commits, and pushes.
                                Used by the GitHub Actions workflow.
"""

import os
import sys
import json
import random
import subprocess
import datetime
from pathlib import Path

import stat
import shutil
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GH_PAT = os.environ.get("GH_PAT")
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "false"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

WORKDIR = Path("workspace")
CONFIG_PATH = Path(__file__).parent / "config.json"

TASK_PROMPTS = {
    "docstring": (
        "You add clear, accurate docstrings to Python functions and classes "
        "that are missing them. Keep all existing code and behavior exactly "
        "the same. Return ONLY the complete updated file content - no "
        "markdown fences, no explanation, nothing before or after the code."
    ),
    "comment_cleanup": (
        "You add brief, useful inline comments to clarify non-obvious logic "
        "in this Python file. Keep all existing code and behavior exactly "
        "the same. Return ONLY the complete updated file content - no "
        "markdown fences, no explanation, nothing before or after the code."
    ),
    "readme_update": (
        "You improve a project README by clarifying or slightly expanding "
        "one section (usage, troubleshooting, or setup). Keep the overall "
        "structure and tone. Return ONLY the complete updated README "
        "content - no markdown fences, no explanation."
    ),
    "lint_fix": (
        "You fix minor style issues in this Python file (unused imports, "
        "inconsistent spacing, overly long lines) without changing "
        "behavior. Return ONLY the complete updated file content - no "
        "markdown fences, no explanation."
    ),
}


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def pick_repo_and_task(config, day_index):
    repo = config["repos"][day_index % len(config["repos"])]
    task = config["tasks"][day_index % len(config["tasks"])]
    return repo, task


def _force_remove(func, path, _exc):
    """git marks some internal files read-only; clear that flag and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree(path):
    try:
        shutil.rmtree(path, onexc=_force_remove)        # Python 3.12+
    except TypeError:
        shutil.rmtree(path, onerror=_force_remove)       # Python < 3.12


def clone_repo(repo, use_auth):
    WORKDIR.mkdir(exist_ok=True)
    name = repo["url"].rstrip("/").split("/")[-1]
    dest = WORKDIR / name
    if dest.exists():
        _rmtree(dest)

    if use_auth and GH_PAT:
        url = repo["url"].replace("https://", f"https://x-access-token:{GH_PAT}@")
    else:
        url = repo["url"]  # public read-only clone, no token needed

    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return dest


def find_python_files(repo_path):
    skip_dirs = {"__pycache__", "venv", ".venv", "node_modules", ".git"}
    files = []
    for p in repo_path.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        files.append(p)
    return files


def pick_target_file(repo_path, task, day_index):
    if task == "readme_update":
        readme = repo_path / "README.md"
        return readme if readme.exists() else None

    candidates = [
        f for f in find_python_files(repo_path)
        if "test" not in f.name.lower() and f.stat().st_size < 20000
    ]
    if not candidates:
        return None

    # Deterministic per day, but varies which file gets picked across days.
    random.seed(day_index)
    random.shuffle(candidates)
    return candidates[0]


def call_groq(system_prompt, user_prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_change(target_file, task):
    original = target_file.read_text(encoding="utf-8")
    updated = call_groq(TASK_PROMPTS[task], f"File: {target_file.name}\n\n{original}")
    return original, updated.strip()


def sanity_check_size(original, updated):
    """Reject changes that look like truncation or runaway repetition -
    a common failure mode where the model still returns valid code, but
    has silently dropped most of the file (or duplicated content)."""
    if not original.strip():
        return True
    ratio = len(updated) / max(len(original), 1)
    return 0.4 <= ratio <= 2.5


def count_flake8_issues(path):
    result = subprocess.run(
        [sys.executable, "-m", "flake8", str(path)],
        capture_output=True, text=True,
    )
    return len(result.stdout.strip().splitlines())


def validate_python_file(path):
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True, text=True,
    )
    return result.returncode == 0, result.stderr


def write_log_entry(repo_name, task, filename, summary):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    month_file = log_dir / f"{datetime.date.today():%Y-%m}.md"
    entry = (
        f"\n### {datetime.date.today():%Y-%m-%d} - {repo_name}\n"
        f"- Task: {task}\n"
        f"- File: {filename}\n"
        f"- Summary: {summary}\n"
    )
    with open(month_file, "a", encoding="utf-8") as f:
        f.write(entry)
    return month_file


def git_commit_and_push(repo_path, message):
    subprocess.run(["git", "-C", str(repo_path), "config", "user.email",
                     "agent@users.noreply.github.com"], check=True)
    subprocess.run(["git", "-C", str(repo_path), "config", "user.name",
                     "ai-commit-agent"], check=True)
    subprocess.run(["git", "-C", str(repo_path), "add", "-A"], check=True)
    result = subprocess.run(
        ["git", "-C", str(repo_path), "commit", "-m", message],
        capture_output=True, text=True,
    )
    if "nothing to commit" in result.stdout:
        print("Nothing changed, skipping push.")
        return False
    subprocess.run(["git", "-C", str(repo_path), "push"], check=True)
    return True


def main():
    if not GROQ_API_KEY:
        sys.exit("Missing GROQ_API_KEY")
    if not DRY_RUN and not GH_PAT:
        sys.exit("Missing GH_PAT for a non-dry-run")

    config = load_config()
    # DAY_OVERRIDE lets you test different repo/task combinations locally
    # without waiting for the real date to roll over, e.g.:
    #   DAY_OVERRIDE=3 python agent.py
    day_index = int(os.environ.get("DAY_OVERRIDE", datetime.date.today().toordinal()))
    repo, task = pick_repo_and_task(config, day_index)
    print(f"Selected repo: {repo['url']}")
    print(f"Selected task: {task}")

    repo_path = clone_repo(repo, use_auth=not DRY_RUN)

    target_file = pick_target_file(repo_path, task, day_index)
    if target_file is None:
        print("No suitable file found for this task today, skipping.")
        return
    print(f"Target file: {target_file}")

    original, updated = generate_change(target_file, task)
    if updated.strip() == original.strip():
        print("Model returned no real change, skipping.")
        return

    print("\n----- PROPOSED CHANGE (first 1000 chars) -----")
    print(updated[:1000])
    print("----- END PREVIEW -----\n")

    if DRY_RUN:
        print("DRY_RUN is on: nothing was written, validated, or committed.")
        return

    if not sanity_check_size(original, updated):
        print("Proposed change size looks suspicious (too short or too "
              "long vs. original) - skipping rather than risk gutting the file.")
        return

    pre_lint_issues = None
    if task == "lint_fix" and target_file.suffix == ".py":
        pre_lint_issues = count_flake8_issues(target_file)

    target_file.write_text(updated, encoding="utf-8")

    if target_file.suffix == ".py":
        ok, err = validate_python_file(target_file)
        if not ok:
            print("Syntax check failed, reverting change:\n", err)
            target_file.write_text(original, encoding="utf-8")
            return

    if pre_lint_issues is not None:
        post_lint_issues = count_flake8_issues(target_file)
        if post_lint_issues > pre_lint_issues:
            print(f"Lint fix made things worse ({pre_lint_issues} -> "
                  f"{post_lint_issues} issues), reverting.")
            target_file.write_text(original, encoding="utf-8")
            return

    summary = f"Applied '{task}' to {target_file.name}"
    pushed = git_commit_and_push(
        repo_path, f"{task}: AI-assisted update to {target_file.name}"
    )

    if pushed:
        write_log_entry(repo["url"].rstrip("/").split("/")[-1], task,
                         target_file.name, summary)
        git_commit_and_push(
            Path("."), f"log: {task} on {repo['url'].rstrip('/').split('/')[-1]}"
        )


if __name__ == "__main__":
    main()