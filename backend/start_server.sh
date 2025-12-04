#!/usr/bin/env bash
cd /Users/basiakoch/bk489/C1_coursework/backend
source .venv/bin/activate
python -m uvicorn fivedreg.main:app --host 0.0.0.0 --port 8000 --reload
