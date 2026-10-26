# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for skill evaluation grading."""

import argparse
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

import pytest

from argparser import omniarg_parser
from rocprof_compute_base import RocProfCompute
from skills.run_evals import grade

USAGE_ERROR = (
    "usage: rocprof-compute [mode] [options]\n"
    "tool: error: unrecognized arguments: --membw-analysis"
)
COMMAND_FAILURE = ["rocprof-compute command failed with a usage error"]
HELP_OUTPUT = "usage: rocprof-compute [mode] [options]\noptions:\n  -h, --help"
SKILL = "rocprof-compute-memory"


def judge_reply(
    judgment: SimpleNamespace, options: List[str], prompt: str, **_kwargs
) -> str:
    """Record the external judge request and return a controlled wire response."""
    judgment.prompts.append(prompt)
    judgment.options.append(options)
    if isinstance(judgment.reply, str):
        return judgment.reply
    return next(judgment.reply)


@pytest.fixture
def usage_judge(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Keep the real judgment decoder while replacing the Claude subprocess."""
    judgment = SimpleNamespace(
        reply=json.dumps({"pass": False, "reason": "An executed CLI command failed."}),
        prompts=[],
        options=[],
    )
    monkeypatch.setattr("skills.run_evals._run_claude", partial(judge_reply, judgment))
    return judgment


def tool_result(
    output: Union[str, List[Dict[str, str]]],
    is_error: bool = False,
    tool_name: str = "Bash",
    command: str = "rocprof-compute analyze --membw-analysis -p workload/",
) -> str:
    """Serialize a tool call and its result as stream-json transcript events."""
    command_id = f"test-{tool_name}-{command}"
    call = {
        "type": "tool_use",
        "name": tool_name,
        "id": command_id,
        "input": {"command": command},
    }
    response = {
        "type": "tool_result",
        "tool_use_id": command_id,
        "content": output,
        "is_error": is_error,
    }
    return "\n".join([
        json.dumps({"type": "assistant", "message": {"content": [call]}}),
        json.dumps({"type": "user", "message": {"content": [response]}}),
    ])


@pytest.mark.parametrize(
    ("transcript", "behavioral", "expected_failures"),
    [
        pytest.param(
            tool_result(USAGE_ERROR, is_error=True), False, [], id="routing-mode"
        ),
        pytest.param(
            tool_result(HELP_OUTPUT, is_error=True)
            + "\n"
            + tool_result("tool: error: unrelated failure", is_error=True),
            True,
            [],
            id="usage-and-error-in-separate-results",
        ),
        pytest.param(
            json.dumps({
                "message": {"content": [{"type": "text", "text": USAGE_ERROR}]}
            }),
            True,
            [],
            id="assistant-explanation",
        ),
        pytest.param(
            tool_result(HELP_OUTPUT)
            + "\n"
            + tool_result(USAGE_ERROR, tool_name="Read"),
            True,
            [],
            id="help-then-read-of-error-example",
        ),
        pytest.param(
            tool_result(
                "usage: rocprof-compute [mode] [options]\n"
                + "        [--example value]\n" * 10
                + "options:\n  -h, --help\ntool: error: unrelated failure",
                command=(
                    "rocprof-compute --help; "
                    "printf '%s\\n' 'tool: error: unrelated failure'"
                ),
            ),
            True,
            [],
            id="help-then-printed-error-in-one-result",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "source /tmp/setup.sh  # ROCm environment\n"
                    "if time timeout -k 5 60s "
                    "'/opt/rocm/bin/rocprof-compute' "
                    "analyze --membw-analysis -p workload/ "
                    "2>&1 | tee analyze.log; then echo done; fi"
                ),
            )
            + "\n"
            + tool_result(
                "3. Memory Chart\n30. Guidance",
                command="rocprof-compute analyze -p workload/ -b 3 30",
            ),
            True,
            COMMAND_FAILURE,
            id="masked-usage-error-then-recovery",
        ),
        pytest.param(
            tool_result(
                [{"type": "text", "text": line} for line in USAGE_ERROR.splitlines()],
                is_error=True,
                command=(
                    "env -i ROCM_PATH='/opt/ROCm SDK' "
                    "python3 -u ./src/rocprof-compute "
                    "analyze --membw-analysis -p workload/"
                ),
            ),
            True,
            COMMAND_FAILURE,
            id="list-based-tool-result",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command='result="$(rocprof-compute analyze --membw-analysis)"',
            ),
            True,
            COMMAND_FAILURE,
            id="quoted-command-substitution",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command='echo "$(echo "$(rocprof-compute analyze --membw-analysis)")"',
            ),
            True,
            COMMAND_FAILURE,
            id="nested-command-substitution",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "result=\"$(cat <<'EOF'\n"
                    "$(rocprof-compute analyze --membw-analysis)\n"
                    f'{USAGE_ERROR}\nEOF\n)"; printf \'%s\\n\' "$result"'
                ),
            ),
            True,
            [],
            id="substitution-prints-quoted-heredoc",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "# rocprof-compute analyze --bad\n"
                    f"printf '%s\\n' x'; rocprof-compute' '{USAGE_ERROR}'"
                ),
            ),
            True,
            [],
            id="bash-prints-error-example",
        ),
        pytest.param(
            tool_result(
                "rocprof-compute analyze --membw-analysis\n" + USAGE_ERROR,
                command=(
                    "cat <<'EOF'\nrocprof-compute analyze --membw-analysis\n"
                    f"{USAGE_ERROR}\nEOF\n"
                ),
            ),
            True,
            [],
            id="bash-prints-quoted-heredoc",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "cat <<EOF\nrocprof-compute analyze --membw-analysis\n"
                    f"{USAGE_ERROR}\nEOF\n"
                ),
            ),
            True,
            [],
            id="bash-prints-unquoted-heredoc",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "cat <<EOF\n# '$(rocprof-compute analyze --membw-analysis)'\nEOF\n"
                ),
            ),
            True,
            COMMAND_FAILURE,
            id="unquoted-heredoc-executes-substitution",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "cat <<-'FIRST' <<'SECOND'\n\tprinted ' example \\\n\tFIRST\n"
                    f"{USAGE_ERROR}\nSECOND\n"
                    "rocprof-compute analyze --membw-analysis"
                ),
            ),
            True,
            COMMAND_FAILURE,
            id="command-after-multiple-heredocs",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "printf '%s\\n' 'example:\ncat <<EOF'; "
                    "rocprof-compute analyze --membw-analysis\n"
                    "# cat <<EOF\n"
                ),
            ),
            True,
            COMMAND_FAILURE,
            id="quoted-and-commented-heredoc-markers",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=("count=$((1 << 2))\nrocprof-compute analyze --membw-analysis"),
            ),
            True,
            COMMAND_FAILURE,
            id="arithmetic-shift-before-command",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "command -v rocprof-compute; cat /tmp/rocprof-compute-errors.log"
                ),
            ),
            True,
            [],
            id="bash-reads-old-error-log",
        ),
        pytest.param(
            tool_result(
                USAGE_ERROR,
                command=(
                    "EXAMPLE='$(rocprof-compute analyze --membw-analysis)'; "
                    'ESCAPED="\\$(rocprof-compute analyze --membw-analysis)"; '
                    f'DIAGNOSTIC="{USAGE_ERROR}"; printf "%s\\n" "$DIAGNOSTIC"'
                ),
            ),
            True,
            [],
            id="bash-prints-quoted-assignment",
        ),
    ],
)
def test_grade_command_usage_errors(
    transcript: str,
    behavioral: bool,
    expected_failures: List[str],
    tmp_path: Path,
    usage_judge: SimpleNamespace,
) -> None:
    failures = grade(
        {"skill_should_trigger": True}, SKILL, transcript, [SKILL], tmp_path, behavioral
    )

    assert failures == expected_failures
    assert bool(usage_judge.prompts) == bool(expected_failures)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(
            ["analyze", "--experimental", "--membw-analysis", "-p", "workload/"],
            id="unknown-analyze-option",
        ),
        pytest.param(
            ["analyze", "--decimal", "abc", "-p", "workload/"],
            id="multiline-analyze-usage",
        ),
        pytest.param(["profile", "--name"], id="multiline-profile-usage"),
    ],
)
def test_grade_real_parser_usage_errors(
    arguments: List[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    usage_judge: SimpleNamespace,
) -> None:
    parser = argparse.ArgumentParser(
        prog="tool", usage="rocprof-compute [mode] [options]"
    )
    omniarg_parser(
        parser, tmp_path, {"gfx950": "MI350"}, {"ver_pretty": "unit test"}, True
    )
    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)
    assert error.value.code == 2
    transcript = tool_result(
        capsys.readouterr().err,
        is_error=True,
        command=" ".join(["rocprof-compute", *arguments]),
    )

    failures = grade(
        {"skill_should_trigger": True}, SKILL, transcript, [SKILL], tmp_path, True
    )

    assert failures == COMMAND_FAILURE


def test_grade_preliminary_parser_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    usage_judge: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["rocprof-compute", "profile", "--experimental=true", "--name", "test"],
    )
    # This error exits before the full parser or GPU setup is reached.
    compute = RocProfCompute.__new__(RocProfCompute)
    with pytest.raises(SystemExit) as error:
        compute.parse_args()
    assert error.value.code == 2
    transcript = tool_result(
        capsys.readouterr().err,
        is_error=True,
        command=(
            "bash -lc 'cd /tmp' && "
            "bash -lc 'rocprof-compute profile --experimental=true --name test'"
        ),
    )

    failures = grade(
        {"skill_should_trigger": True}, SKILL, transcript, [SKILL], tmp_path, True
    )

    assert failures == COMMAND_FAILURE


@pytest.mark.parametrize(
    ("command", "output", "reply", "expected_failures"),
    [
        pytest.param(
            f"rocprof-compute --version; printf '%s\\n' '{USAGE_ERROR}'",
            "version 1.0\n" + USAGE_ERROR,
            {"pass": True, "reason": "Only printf prints the diagnostic."},
            [],
            id="version-then-printed-diagnostic",
        ),
        pytest.param(
            f"rocprof-compute --version; cat <<'EOF'\n{USAGE_ERROR}\nEOF\n",
            "version 1.0\n" + USAGE_ERROR,
            {"pass": True, "reason": "Only cat prints the literal heredoc."},
            [],
            id="version-then-heredoc-diagnostic",
        ),
        pytest.param(
            "rocprof-compute analyze -p workload/ -b 3 30; "
            f"printf '%s\\n' '{USAGE_ERROR}'",
            "3. Memory Chart\n30. Guidance\n" + USAGE_ERROR,
            {"pass": True, "reason": "Analysis succeeds; printf prints the example."},
            [],
            id="analysis-then-printed-diagnostic",
        ),
        pytest.param(
            f"rocprof-compute analyze --membw-analysis; printf '%s\\n' '{USAGE_ERROR}'",
            USAGE_ERROR + "\n" + USAGE_ERROR,
            {"pass": False, "reason": "The first diagnostic is a real CLI failure."},
            COMMAND_FAILURE,
            id="real-error-and-identical-printed-example",
        ),
        pytest.param(
            "rocprof-compute analyze --membw-analysis && "
            f"printf '%s\\n' '{USAGE_ERROR}'",
            USAGE_ERROR,
            {"pass": False, "reason": "The failing CLI prevents printf from running."},
            COMMAND_FAILURE,
            id="real-error-and-skipped-printer",
        ),
        pytest.param(
            "rocprof-compute --version; "
            "rocprof-compute analyze --membw-analysis || true",
            "version 1.0\n" + USAGE_ERROR,
            {"pass": False, "reason": "The version output does not clear a failure."},
            COMMAND_FAILURE,
            id="version-then-masked-real-error",
        ),
        pytest.param(
            "rocprof-compute analyze --membw-analysis || true; "
            "rocprof-compute --version",
            USAGE_ERROR + "\nversion 1.0",
            {"pass": False, "reason": "Later success does not clear a failure."},
            COMMAND_FAILURE,
            id="masked-real-error-then-version",
        ),
        pytest.param(
            "rocprof-compute analyze --membw-analysis",
            USAGE_ERROR,
            "unparseable response",
            COMMAND_FAILURE,
            id="unparseable-usage-judgment-fails-closed",
        ),
        pytest.param(
            "rocprof-compute analyze --membw-analysis",
            USAGE_ERROR,
            '{"pass": "false", "reason": "Not a boolean verdict."}',
            COMMAND_FAILURE,
            id="invalid-usage-verdict-fails-closed",
        ),
        pytest.param(
            "rocprof-compute analyze --membw-analysis",
            USAGE_ERROR,
            "[]",
            COMMAND_FAILURE,
            id="invalid-usage-judge-payload-fails-closed",
        ),
    ],
)
def test_grade_correlates_usage_diagnostics(
    command: str,
    output: str,
    reply: Union[str, Dict[str, Union[bool, str]]],
    expected_failures: List[str],
    tmp_path: Path,
    usage_judge: SimpleNamespace,
) -> None:
    usage_judge.reply = reply if isinstance(reply, str) else json.dumps(reply)
    transcript = tool_result(output, command=command)
    # A different tool result must not enter this candidate's judgment context.
    transcript += "\n" + tool_result("unrelated file contents", tool_name="Read")

    failures = grade(
        {"skill_should_trigger": True},
        SKILL,
        transcript,
        [SKILL],
        tmp_path,
        behavioral=True,
    )

    assert failures == expected_failures
    assert len(usage_judge.prompts) == 1
    assert usage_judge.options[0][-2:] == ["--tools", ""]
    context = usage_judge.prompts[0].split("Transcript:\n", 1)[1]
    assert json.loads(context) == {"command": command, "output": output}


def test_grade_keeps_usage_failure_when_judge_context_would_be_truncated(
    tmp_path: Path, usage_judge: SimpleNamespace
) -> None:
    command = (
        f"rocprof-compute --version; printf '%s\\n' '{USAGE_ERROR}'; "
        "python3 -c \"print('x' * 210000)\"; "
        "rocprof-compute analyze --membw-analysis || true"
    )
    output = "version 1.0\n" + USAGE_ERROR + "\n" + "x" * 210000 + "\n" + USAGE_ERROR
    usage_judge.reply = json.dumps({
        "pass": True,
        "reason": "Only the example is seen.",
    })

    failures = grade(
        {"skill_should_trigger": True},
        SKILL,
        tool_result(output, command=command),
        [SKILL],
        tmp_path,
        behavioral=True,
    )

    assert failures == COMMAND_FAILURE
    assert not usage_judge.prompts


@pytest.mark.parametrize(
    ("second_command", "second_pass", "expected_failures"),
    [
        pytest.param(
            f"rocprof-compute --version; cat <<'EOF'\n{USAGE_ERROR}\nEOF\n",
            True,
            [],
            id="multiple-printed-examples",
        ),
        pytest.param(
            "rocprof-compute analyze --membw-analysis || true",
            False,
            COMMAND_FAILURE,
            id="printed-example-then-real-failure",
        ),
    ],
)
def test_grade_checks_later_usage_candidates(
    second_command: str,
    second_pass: bool,
    expected_failures: List[str],
    tmp_path: Path,
    usage_judge: SimpleNamespace,
) -> None:
    first_command = f"rocprof-compute --version; printf '%s\\n' '{USAGE_ERROR}'"
    transcript = tool_result("version 1.0\n" + USAGE_ERROR, command=first_command)
    second_output = "version 1.0\n" + USAGE_ERROR if second_pass else USAGE_ERROR
    transcript += "\n" + tool_result(second_output, command=second_command)
    usage_judge.reply = iter([
        json.dumps({"pass": True, "reason": "The first diagnostic is an example."}),
        json.dumps({"pass": second_pass, "reason": "The next invocation is assessed."}),
    ])

    failures = grade(
        {"skill_should_trigger": True},
        SKILL,
        transcript,
        [SKILL],
        tmp_path,
        behavioral=True,
    )

    assert failures == expected_failures
    contexts = [prompt.split("Transcript:\n", 1)[1] for prompt in usage_judge.prompts]
    assert [json.loads(context)["command"] for context in contexts] == [
        first_command,
        second_command,
    ]
