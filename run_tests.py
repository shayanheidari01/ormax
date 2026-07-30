#!/usr/bin/env python3
"""
Test runner script for Ormax ORM.
Executes pytest suite and outputs results.
"""
import sys
import os
import io

# Set UTF-8 encoding for standard output on Windows
if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

import pytest

def main():
    print("=" * 60)
    print("Running Ormax ORM Test Suite")
    print("=" * 60)
    
    args = ["-v", "tests/"]
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
        
    exit_code = pytest.main(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
