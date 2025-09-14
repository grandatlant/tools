#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "",
# ]
# ///
"""Doc
"""

import sys


##  MAIN ENTRY POINT
def main(args=None):
    args = sys.argv if args is None else args
    print(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
