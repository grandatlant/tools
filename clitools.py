"""Tools to simplify work with CLI
"""

__version__ = '1.0.1'
__copyright__ = 'Copyright (C) 2025 grandatlant'

__all__ = [
    'confirm_action',
    'readlines',
]


def confirm_action(prompt, /, confirmations = {'y','yes'}):
    return (input(prompt).strip().lower() in confirmations)


def readlines(prompt = None, lines = None, end = None):
    """Generator for input() until EOFError"""
    prompt_input = str(prompt) if prompt else ''
    append_lines = bool(isinstance(lines, list))
    
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
