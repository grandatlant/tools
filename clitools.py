"""Tools to simplify work with CLI
"""

__version__ = '1.0.2'
__copyright__ = 'Copyright (C) 2025 grandatlant'

__all__ = [
    'confirm_action',
    'readlines',
]


from typing import (
    Any,
    Optional,
    runtime_checkable,
    Protocol,
    Container,
    Generator,
)


@runtime_checkable
class Appendable(Protocol):
    def append(self, item: Any) -> None: ...


def confirm_action(
    prompt: str,
    confirmations: Container[str] = {'y','yes'}
) -> bool:
    """Accept confirmation from user in form "y" or "yes"
    or any in "confirmations" given in lowercase.
    """
    return (input(prompt).strip().lower() in confirmations)


def readlines(
    prompt: Optional[str] = None,
    lines: Optional[Appendable] = None,
    end: Optional[str] = None,
) -> Generator[str, None, None]:
    """Generator for input() until EOFError (Ctrl+D) or KeyboardInterrupt.
    Params:
        "prompt" will be used for input() if given.
        "lines" is list or object with 'append' method
            to append lines returned with input()
        "end" if given, will be appended to "lines"
            if specified, and then yield before StopIteration.
    """
    prompt_input = str(prompt) if prompt else ''
    append_lines = False if lines is None else isinstance(lines, Appendable)
    while True:
        try:
            line = input(prompt_input)
        except (EOFError, KeyboardInterrupt):
            break
        if append_lines:
            lines.append(line)
        yield line
    if end is not None:
        if append_lines:
            lines.append(end)
        yield end
