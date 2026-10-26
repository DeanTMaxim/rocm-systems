#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Run the skill evaluation datasets through Claude Code.

There is no CI for skills, so this is the manual gate: run it before a release
to check that each skill still routes on the prompts it should and stays quiet
on the ones it should not.

Usage:
    ./skills/run_evals.py                      # routing only, no tools
    ./skills/run_evals.py --mode behavioral    # also grade what the agent did
    ./skills/run_evals.py --skill rocprof-compute-memory  # one skill
    ./skills/run_evals.py --list               # validate datasets, run nothing

Exits 0 when every case passes, 1 on a failure, and 77 when the `claude` CLI
is missing so a caller can treat that as skipped rather than broken.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Tuple

SKILLS_ROOT = Path(__file__).resolve().parent
SAMPLES_ROOT = SKILLS_ROOT.parent / "sample"
EXIT_SKIPPED = 77

# Only these keys are understood. A typo is an error rather than an expectation
# that silently never runs.
KNOWN_KEYS = {
    "id",
    "prompt",
    "skill_should_trigger",
    "note",
    "expected_behavior",
    "unexpected_behavior",
    "logs_contain",
    "files_exist",
    "sample",
}
JUDGED_KEYS = ("expected_behavior", "unexpected_behavior")
BEHAVIORAL_KEYS = (*JUDGED_KEYS, "logs_contain", "files_exist", "sample")

# Keep the agent off the user's own CLAUDE.md, skills, and settings.
ISOLATED_SETTINGS = ["--setting-sources", "project,local"]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["pass", "reason"],
}


def discover_skills(only: Optional[str]) -> list[Path]:
    """Return skill directories that ship an evals dataset."""
    found = sorted(p.parent.parent for p in SKILLS_ROOT.glob("*/evals/evals.json"))
    if only:
        found = [p for p in found if p.name == only]
        if not found:
            sys.exit(f"No skill named {only!r} with an evals dataset.")
    return found


def load_cases(skill: Path) -> list[dict[str, Any]]:
    """Parse and validate one dataset, erroring on anything malformed."""
    dataset = json.loads((skill / "evals" / "evals.json").read_text())
    cases = dataset["evaluations"]

    positive = 0
    negative = 0
    judged = 0
    seen: set[str] = set()

    for case in cases:
        unknown = set(case) - KNOWN_KEYS
        if unknown:
            sys.exit(
                f"{skill.name}: unknown key(s) {sorted(unknown)} in {case.get('id')}"
            )
        if not isinstance(case.get("skill_should_trigger"), bool):
            sys.exit(
                f"{skill.name}: {case.get('id')} needs a boolean skill_should_trigger"
            )
        if case["id"] in seen:
            sys.exit(f"{skill.name}: duplicate case id {case['id']!r}")
        seen.add(case["id"])
        if "sample" in case and not (SAMPLES_ROOT / case["sample"]).exists():
            sys.exit(f"{skill.name}: {case['id']} names a missing sample")

        if case["skill_should_trigger"]:
            positive += 1
            judged += any(case.get(key) for key in JUDGED_KEYS)
        else:
            negative += 1
            extra = set(case) - {"id", "prompt", "skill_should_trigger", "note"}
            if extra:
                sys.exit(
                    f"{skill.name}: {case['id']} is negative but sets {sorted(extra)}"
                )

    if positive < 3 or negative < 2:
        sys.exit(f"{skill.name}: needs at least 3 positive and 2 negative cases")
    if not judged:
        sys.exit(f"{skill.name}: needs at least one positive case with judged behavior")

    return cases


def build_workspace(root: Path) -> Path:
    """Install every skill and copy the samples into a scratch workspace."""
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    subprocess.run(
        [str(SKILLS_ROOT / "install-skills.sh"), "--agent", "claude"],
        env={**os.environ, "HOME": str(workspace)},
        stdout=subprocess.DEVNULL,
        check=True,
    )
    shutil.copytree(SAMPLES_ROOT, workspace / "sample")
    return workspace


def case_prompt(case: dict[str, Any]) -> str:
    """Return the prompt, pointing at the case's sample when it has one."""
    if "sample" not in case:
        return case["prompt"]
    return f"{case['prompt']}\n\nThe application is ./sample/{case['sample']}."


def run_agent(prompt: str, workspace: Path, behavioral: bool) -> tuple[str, list[str]]:
    """Run one prompt and return the transcript text and the skills it loaded.

    Raises RuntimeError when the claude CLI itself fails.
    """
    options = [
        "--output-format",
        "stream-json",
        "--verbose",
        "--add-dir",
        str(workspace),
    ]
    if behavioral:
        # The agent has to build and profile the sample, so it runs commands
        # without asking.
        options += ["--permission-mode", "bypassPermissions"]
    else:
        # Routing only: the decision to load a skill is all we grade, so deny
        # the agent the tools that would make it start doing the work.
        options += ["--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit"]

    output = _run_claude(
        options, prompt, cwd=workspace, timeout=900 if behavioral else 300
    )

    transcript: list[str] = []
    loaded: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        transcript.append(line)
        loaded.extend(_skills_in(event))

    return "\n".join(transcript), loaded


def _skills_in(event: Any) -> list[str]:
    """Pull skill names out of Skill tool calls and SKILL.md reads.

    Events of any other shape are skipped, so new CLI event types are ignored.
    """
    names: list[str] = []
    message = event.get("message") if isinstance(event, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return names

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        payload = block.get("input")
        if not isinstance(payload, dict):
            continue
        if block.get("name") == "Skill":
            skill = str(payload.get("skill", ""))
            names.append(skill.split(":")[-1])
        elif block.get("name") == "Read":
            path = str(payload.get("file_path", ""))
            if path.endswith("SKILL.md"):
                names.append(Path(path).parent.name)
    return names


def _run_claude(
    options: list[str], prompt: str, cwd: Optional[Path] = None, timeout: int = 300
) -> str:
    """Run claude -p with the prompt on stdin and return its stdout.

    Raises RuntimeError with the CLI's own error text when it exits non-zero.
    """
    result = subprocess.run(
        ["claude", "-p", *options, *ISOLATED_SETTINGS],
        input=prompt,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        reason = _error_text(result.stdout) or result.stderr.strip()[-500:]
        raise RuntimeError(f"claude exited {result.returncode}: {reason}")
    return result.stdout


def _error_text(output: str) -> str:
    """Return the result text of the last JSON event in claude output, if any."""
    for line in reversed(output.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            return str(event.get("result", ""))
    return ""


def grade_judged(claim: str, transcript: str, should_hold: bool) -> tuple[bool, str]:
    """Ask a second agent whether a plain-language claim holds in the transcript.

    Raises RuntimeError when the claude CLI itself fails.
    """
    question = (
        "You are grading an AI agent transcript against one claim.\n"
        f"Claim: {claim}\n"
        "Set pass to true if the claim holds in the transcript, false if not.\n"
        "Answer only with the JSON object.\n\n"
        f"Transcript:\n{transcript[:200000]}"
    )
    output = _run_claude(
        ["--output-format", "json", "--json-schema", json.dumps(JUDGE_SCHEMA)],
        question,
    )
    try:
        payload = json.loads(output)
        verdict = payload if "pass" in payload else json.loads(payload["result"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, "judge returned unparseable output"

    held = bool(verdict.get("pass"))
    return held == should_hold, str(verdict.get("reason", ""))


def grade(
    case: dict[str, Any],
    skill: str,
    transcript: str,
    loaded: list[str],
    workspace: Path,
    behavioral: bool,
) -> list[str]:
    """Return the list of failure messages for one case, empty when it passes."""
    failures: list[str] = []

    triggered = skill in loaded
    if triggered != case["skill_should_trigger"]:
        want = "trigger" if case["skill_should_trigger"] else "not trigger"
        failures.append(f"expected {skill} to {want}, loaded={loaded or 'none'}")

    if not behavioral:
        return failures

    if _has_command_usage_error(transcript):
        failures.append("rocprof-compute command failed with a usage error")

    for needle in case.get("logs_contain", []):
        if needle not in transcript:
            failures.append(f"logs missing {needle!r}")

    for artifact in case.get("files_exist", []):
        if not any(p.match(f"**/{artifact}") for p in workspace.rglob("*")):
            failures.append(f"missing artifact {artifact!r}")

    for claim in case.get("expected_behavior", []):
        ok, reason = grade_judged(claim, transcript, should_hold=True)
        if not ok:
            failures.append(f"expected behavior not met: {claim} ({reason})")

    for claim in case.get("unexpected_behavior", []):
        ok, reason = grade_judged(claim, transcript, should_hold=False)
        if not ok:
            failures.append(f"unexpected behavior occurred: {claim} ({reason})")

    return failures


def run_case(
    name: str, case: dict[str, Any], tmp: Path, behavioral: bool
) -> tuple[str, str, list[str]]:
    """Run and grade one case. Returns (skill, case id, failure messages)."""
    workspace = build_workspace(tmp / f"{name}-{case['id']}")
    try:
        transcript, loaded = run_agent(case_prompt(case), workspace, behavioral)
        failures = grade(case, name, transcript, loaded, workspace, behavioral)
    except subprocess.TimeoutExpired:
        return name, case["id"], ["timed out"]
    except RuntimeError as error:
        return name, case["id"], [str(error)]
    return name, case["id"], failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["routing", "behavioral"], default="routing")
    parser.add_argument("--skill", help="Run one skill instead of all of them.")
    parser.add_argument("--list", action="store_true", help="Validate datasets only.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Cases to run concurrently (DEFAULT: 8). Use 1 to run serially.",
    )
    args = parser.parse_args()

    skills = discover_skills(args.skill)
    datasets = {skill.name: load_cases(skill) for skill in skills}

    if args.list:
        for name, cases in datasets.items():
            positive = sum(case["skill_should_trigger"] for case in cases)
            print(f"{name}: {len(cases)} cases ({positive} positive)")
        return 0

    if not shutil.which("claude"):
        print("SKIP: the claude CLI is not installed; skill evals not run.")
        return EXIT_SKIPPED

    behavioral = args.mode == "behavioral"
    if not behavioral:
        for cases in datasets.values():
            for case in cases:
                for key in BEHAVIORAL_KEYS:
                    case.pop(key, None)

    queued = [(name, case) for name, cases in datasets.items() for case in cases]
    started = time.monotonic()
    results: dict[str, list[tuple[str, list[str]]]] = {name: [] for name in datasets}

    # Each case gets its own workspace and its own agent, so they are
    # independent. The work is waiting on a subprocess, not on Python.
    with tempfile.TemporaryDirectory() as tmp:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                pool.submit(run_case, name, case, Path(tmp), behavioral): name
                for name, case in queued
            }
            for done in as_completed(futures):
                name, case_id, failures = done.result()
                results[name].append((case_id, failures))
                print("." if not failures else "x", end="", flush=True)

    print()
    failed = 0
    for name, cases in sorted(results.items()):
        print(f"\n=== {name} ===")
        for case_id, failures in sorted(cases):
            if failures:
                failed += 1
                print(f"FAIL {case_id}")
                for failure in failures:
                    print(f"     {failure}")
            else:
                print(f"PASS {case_id}")

    elapsed = time.monotonic() - started
    total = len(queued)
    print(f"\n{total - failed}/{total} passed in {elapsed:.0f}s ({args.mode} mode)")
    return 1 if failed else 0


def _has_command_usage_error(transcript: str) -> bool:
    """Detect a CLI usage error within one Bash command's output."""
    commands = {}
    for event_line in transcript.splitlines():
        try:
            event = json.loads(event_line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            command_id = block.get("id")
            if (
                block.get("type") == "tool_use"
                and block.get("name") == "Bash"
                and isinstance(command_id, str)
            ):
                payload = block.get("input")
                command = payload.get("command") if isinstance(payload, dict) else None
                commands[command_id] = command if isinstance(command, str) else ""
                continue
            if (
                block.get("type") != "tool_result"
                or block.get("tool_use_id") not in commands
            ):
                continue
            output = _tool_result_text(block.get("content"))
            # Subparser usage includes examples, but help's option sections
            # must not bridge a usage line to an unrelated printed error.
            if re.search(
                r"^usage:\s*`?rocprof-compute\b[^\n]*\n"
                r"(?:[ \t][^\n]*\n|\n)*"
                r"(?:^-+\nExamples:\n(?:[ \t][^\n]*\n)*"
                r"^-+\n(?:[ \t]*\n)*)?"
                r"^(?:tool|rocprof-compute): error:",
                output,
                re.MULTILINE,
            ) and _invokes_rocprof_compute(commands[block["tool_use_id"]]):
                return True
    return False


def _tool_result_text(content: Any) -> str:
    """Normalize string and text-block tool results for diagnostic matching."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    )


def _invokes_rocprof_compute(command: str) -> bool:
    """Recognize CLI commands following shell keywords and common launchers."""
    command, expanding_bodies = _without_heredoc_bodies(command)
    command = command.replace("\\\n", "")
    if _invokes_command_substitution(command) or any(
        _invokes_command_substitution(body, heredoc=True) for body in expanding_bodies
    ):
        return True
    # Keep complete quoted words intact, including quotes within assignments.
    # Comments exclude their newline so the next command remains visible.
    words = re.findall(
        r"""#[^\n]*|(?:[^\s;&|()\\'"]|\\.|'[^']*'|"(?:\\.|[^"\\])*")+"""
        r"|[;&|()\n]+",
        command,
    )
    prefixes = {
        "env",
        "command",
        "exec",
        "bash",
        "sh",
        "time",
        "!",
        "{",
        "if",
        "then",
        "elif",
        "else",
        "while",
        "until",
        "do",
    }
    operand_options = {
        "env": {"-u", "--unset", "-C", "--chdir"},
        "python": {"-W", "-X"},
        "exec": {"-a"},
        "timeout": {"-s", "--signal", "-k", "--kill-after"},
    }
    expect_executable = True
    launcher = ""
    skip_operand = False
    timeout_duration = False
    try:
        for word in words:
            if word.startswith("#"):
                continue
            if all(character in ";&|()\n" for character in word):
                expect_executable = True
                launcher = ""
                skip_operand = timeout_duration = False
                continue
            if not expect_executable:
                continue
            token = shlex.split(word)[0]
            if skip_operand:
                skip_operand = False
                continue
            if launcher == "shell-command" and _invokes_rocprof_compute(token):
                return True
            if launcher == "shell-command":
                expect_executable = False
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
                continue
            if launcher == "command" and re.fullmatch(r"-[pvV]*[vV][pvV]*", token):
                expect_executable = False
                continue
            if token.startswith("-"):
                skip_operand = token in operand_options.get(launcher, ())
                shell_command = (
                    launcher in {"bash", "sh"}
                    and not token.startswith("--")
                    and "c" in token[1:]
                )
                launcher = "shell-command" if shell_command else launcher
                continue
            if timeout_duration:
                timeout_duration = False
                continue
            executable = Path(token).name
            if executable == "rocprof-compute":
                return True
            if executable == "timeout":
                launcher = executable
                timeout_duration = True
                continue
            python_launcher = bool(re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable))
            launcher = "python" if python_launcher else executable
            expect_executable = executable in prefixes or python_launcher
    except ValueError:
        return False
    return False


def _without_heredoc_bodies(command: str) -> Tuple[str, List[str]]:
    """Separate shell commands from literal heredoc text and expanding bodies."""
    tokens = re.compile(
        r'''\\[\s\S]|'[^']*'|"(?:\\[\s\S]|[^"\\])*"'''
        r"|(?<![^\s;&|()])\#[^\n]*"
        r"|\(\([\s\S]*?\)\)"
        r"""|(?<!<)<<(?!<)(?P<tabs>-?)[ \t]*(?P<delimiter>"""
        r"""(?:[^\s;&|()<>\\'"]|\\.|'[^']*'|"(?:\\.|[^"\\])*")+)"""
        r"|(?P<newline>\n)"
    )
    commands = []
    expanding_bodies = []
    pending: List[Tuple[str, bool, bool]] = []
    cursor = kept_start = 0
    while cursor < len(command):
        token = tokens.search(command, cursor)
        if token is None:
            break
        cursor = token.end()
        delimiter = token.group("delimiter")
        if delimiter is not None:
            pending.append((
                shlex.split(delimiter)[0],
                bool(token.group("tabs")),
                not any(character in delimiter for character in "'\"\\"),
            ))
        if token.group("newline") is None or not pending:
            continue
        commands.append(command[kept_start:cursor])
        for delimiter, strip_tabs, expands in pending:
            closing = re.compile(
                r"^"
                + (r"\t*" if strip_tabs else "")
                + re.escape(delimiter)
                + r"(?:\n|$)",
                re.MULTILINE,
            ).search(command, cursor)
            body_end = closing.start() if closing else len(command)
            if expands:
                expanding_bodies.append(command[cursor:body_end])
            cursor = closing.end() if closing else len(command)
        kept_start = cursor
        pending.clear()
    commands.append(command[kept_start:])
    return "".join(commands), expanding_bodies


def _invokes_command_substitution(command: str, heredoc: bool = False) -> bool:
    """Inspect executable $(...) expansions while preserving shell quoting."""
    # Quotes and comments are literal at the top level of an unquoted heredoc.
    quote = '"' if heredoc else ""
    body_start = parenthesis_depth = index = 0
    substitutions: List[Tuple[str, int, int]] = []

    while index < len(command):
        character = command[index]
        if quote == "'":
            if character == "'":
                quote = ""
            index += 1
            continue
        if character == "\\":
            index += 2
            continue
        if (
            character == "#"
            and not quote
            and (
                index == 0
                or command[index - 1].isspace()
                or command[index - 1] in ";|&("
            )
        ):
            newline = command.find("\n", index)
            index = len(command) if newline == -1 else newline + 1
            continue
        if character == "'" and not quote:
            quote = "'"
        elif character == '"' and (substitutions or not heredoc):
            quote = "" if quote else '"'
        elif command.startswith("$(", index) and not command.startswith("$((", index):
            substitutions.append((quote, body_start, parenthesis_depth))
            quote = ""
            body_start = index + 2
            parenthesis_depth = 1
            index += 2
            continue
        elif substitutions and not quote and character == "(":
            parenthesis_depth += 1
        elif substitutions and not quote and character == ")":
            parenthesis_depth -= 1
            if parenthesis_depth:
                index += 1
                continue
            # Inspect the outer body first so its heredocs are removed before
            # any nested expansions are treated as executable commands.
            if len(substitutions) == 1 and _invokes_rocprof_compute(
                command[body_start:index]
            ):
                return True
            quote, body_start, parenthesis_depth = substitutions.pop()
        index += 1
    return False


if __name__ == "__main__":
    sys.exit(main())
