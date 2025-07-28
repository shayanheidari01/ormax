# test_runner.sh (Bash script for easy testing)
#!/bin/bash

echo "Ormax ORM Test Runner"
echo "===================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed!"
    exit 1
fi

# Run smoke test
echo "Running smoke test..."
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from test_ormax import smoke_test
try:
    result = asyncio.run(smoke_test())
    sys.exit(0 if result else 1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Smoke test failed!"
    exit 1
fi

echo "Smoke test passed!"

# Run full test suite
echo "Running full test suite..."
python3 test_ormax.py

echo "Test run completed!"