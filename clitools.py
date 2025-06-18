"""Tools to simplify work with CLI
"""

__version__ = '1.0.1'
__copyright__ = 'Copyright (C) 2025 grandatlant'

__all__ = [
    'confirm_action',
    'readlines',
]


def confirm_action(prompt, /, confirmations = {'y','yes'}):
    """Accept confirmation from user in form "y" or "yes"
    or any in "confirmations" set given."""
    return (input(prompt).strip().lower() in confirmations)


def readlines(prompt = None, lines = None, end = None):
    """Generator for input() until EOFError (Ctrl+D).
    Params:
        "prompt" will be used for input() if given.
        "lines" is list or object with 'append' method
            to append lines returned with input()
        "end" if given, will be appended to "lines"
            if specified, and then yield before StopIteration
    """
    prompt_input = str(prompt) if prompt else ''
    append_lines = (
        isinstance(lines, list)
        or (hasattr(lines, 'append') and callable(lines.append))
    )
    while True:
        try:
            line = input(prompt_input)
        except EOFError:
            break
        if append_lines:
            lines.append(line)
        yield line
    if end is not None:
        if append_lines:
            lines.append(end)
        yield end
