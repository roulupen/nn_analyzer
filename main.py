"""Main entry point for the Neural Network Analyzer application."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Run the Neural Network Analyzer web application."""
    import uvicorn
    print("Starting Neural Network Analyzer...")
    print("Access the application at: http://localhost:8000")
    uvicorn.run("nn_analyzer.web.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
