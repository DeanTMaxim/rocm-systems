# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for skill evaluation grading."""

import argparse
import json
from pathlib import Path
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
) -> None:
    failures = grade(
        {"skill_should_trigger": True}, SKILL, transcript, [SKILL], tmp_path, behavioral
    )

    assert failures == expected_failures


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
    arguments: List[str], tmp_path: Path, capsys: pytest.CaptureFixture
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
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
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
