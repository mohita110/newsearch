#!/bin/bash
# Setup virtual environment for the newsearch system

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Setup complete."
echo ""
echo "Activate with:  source venv/bin/activate"
echo "Run pipeline:   python run_pipeline.py"
echo "Start API:      uvicorn api:app --host 0.0.0.0 --port 8000"
echo "  or:           python api.py"
