1. Create a venv 
2. Install dependencies
3. Run the app
    cd YOLOv5
    windows: uvicorn app:app --host 127.0.0.1 --port 5000 --reload
    mac: DEVICE=mps uvicorn app:app --host 0.0.0.0 --port 8000 --reload
4. go to http://127.0.0.1:8000/docs
    
