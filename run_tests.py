# run_tests.py (Alternative simple test runner)
"""
Simple test runner for Ormax ORM
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_ormax import run_all_tests, smoke_test

async def main():
    """Main test runner"""
    print("Ormax ORM Test Runner")
    print("=" * 50)
    
    # Run smoke test first
    print("Running smoke test...")
    smoke_success = await smoke_test()
    
    if not smoke_success:
        print("Smoke test failed! Exiting...")
        return False
    
    print("Smoke test passed! Running full test suite...")
    print()
    
    # Run full test suite
    return await run_all_tests()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit_code = 0 if success else 1
    sys.exit(exit_code)