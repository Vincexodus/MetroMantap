# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import uvicorn

app = FastAPI()

@app.get("/predictions/")
async def get_all_predictions():
    base_path = os.path.join("model", "lstm")
    all_predictions = {}

    # Iterate through all folders in the base_path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        file_path = os.path.join(folder_path, "future_predictions.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                predictions = df.to_dict(orient="records")
                all_predictions[folder_name] = predictions
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading file {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")

    if not all_predictions:
        raise HTTPException(status_code=404, detail="No predictions found in any folder")

    return JSONResponse(content=all_predictions)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run(app, host="192.168.1.100", port=8000)