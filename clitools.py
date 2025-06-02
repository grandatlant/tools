#!/usr/bin/env -S python3 -OO
# -*- coding=utf-8 -*-
"""Tools to simplify work with CLI
"""

__version__ = '1.0.0'

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

##  MAIN
def main():
    return 0

if __name__ == '__main__':
    main()
