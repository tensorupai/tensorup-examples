import logging

from supportevals import __main__ as cli
from supportevals.cases import CASES


def asyncio_record(message: str, exc: BaseException) -> logging.LogRecord:
    return logging.LogRecord(
        "asyncio", logging.ERROR, __file__, 1, message, None, (type(exc), exc, None)
    )


def test_stream_close_noise_is_dropped():
    record = asyncio_record(
        "an error occurred during closing of asynchronous generator <async_generator object x>",
        RuntimeError("aclose(): asynchronous generator is already running"),
    )
    assert cli.StreamCloseNoise().filter(record) is False


def test_other_asyncio_errors_still_show():
    other = asyncio_record("Task exception was never retrieved", RuntimeError("boom"))
    assert cli.StreamCloseNoise().filter(other) is True


def test_progress_prints_one_line_per_run(capsys):
    task = cli.with_progress(lambda case: {"output": "ok"}, total=2)
    task(CASES[0])
    task(CASES[1])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"[1/2] {CASES[0].name}")
    assert lines[1].startswith(f"[2/2] {CASES[1].name}")
