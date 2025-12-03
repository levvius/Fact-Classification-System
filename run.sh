#!/bin/bash

# Script to run the Fact Classification API + Frontend

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Starting Fact Classification System..."
echo ""

# Check if port 8000 is already in use
if command -v lsof > /dev/null 2>&1; then
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Warning: Port 8000 is already in use."
        echo "    Another instance may be running."
        echo "    Kill it with: kill \$(lsof -t -i:8000)"
        echo ""
        read -p "    Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Aborted"
            exit 1
        fi
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "   Please ensure Python 3.9+ is installed"
        exit 1
    fi
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install dependencies
echo "✓ Checking dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    echo "   Please check requirements.txt and try again"
    exit 1
fi

# Check if Knowledge Base exists
if [ ! -f "data/faiss_index/wikipedia.index" ]; then
    echo ""
    echo "⚠️  Knowledge Base not found!"
    echo "   Building KB (this will take 2-5 minutes)..."
    echo ""
    python scripts/build_kb.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build Knowledge Base"
        echo "   Please check the error message above"
        exit 1
    fi
    echo ""
    echo "✓ Knowledge Base built successfully"
else
    echo "✓ Knowledge Base found"
fi

# Start the server
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Starting FastAPI server..."
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  📡 API Docs:      http://localhost:8000/docs"
echo "  🌐 Web UI:        http://localhost:8000"
echo "  ❤️  Health Check:  http://localhost:8000/api/v1/health"
echo ""
echo "  Models will load in 5-10 seconds (first request)"
echo "  Press CTRL+C to stop"
echo "════════════════════════════════════════════════════════════"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000
