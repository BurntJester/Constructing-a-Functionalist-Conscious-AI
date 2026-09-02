#!/bin/bash

echo "[Ensemble Boot Sequence Initiated]"

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "[Creating Python Virtual Environment...]"
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install requirements silently
echo "[Verifying Dependencies...]"
pip install -r requirements.txt -q

# Start the FastAPI server in the background
echo "[Igniting Cognitive Loop on Port 8000...]"
uvicorn server:app --reload --port 8000 &

# Wait a few seconds for the server to bind
sleep 3

# Open the UI in the default web browser (OS dependent)
echo "[Launching Interface...]"
if which xdg-open > /dev/null
then
  xdg-open index.html
elif which gnome-open > /dev/null
then
  gnome-open index.html
elif which open > /dev/null
then
  open index.html
fi

echo "[Boot Sequence Complete]"