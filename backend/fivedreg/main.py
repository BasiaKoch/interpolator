from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
import numpy as np, tempfile, os
from fivedreg.data.loader import load_dataset_pkl
from fivedreg.interpolator import train_model, interpolate, save_model
from fivedreg.api.state import STATE

app = FastAPI(title="5D Interpolator")

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health(): return {"status": "ok"}

class TrainRequest(BaseModel):
    batch_size: int = 256
    max_epochs: int = 200
    lr: float = 5e-3
    weight_decay: float = 1e-6
    patience: int = 20

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pkl"):
        raise HTTPException(400, "Expect .pkl file with tuple (X, y)")

    try:
        data = await file.read()
        X, y = load_dataset_pkl(data)
    except Exception as e:
        raise HTTPException(400, f"Failed to load dataset: {str(e)}")

    # Create temp file with proper cleanup
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz", mode='wb') as tmp:
            tmp_path = tmp.name
        np.savez(tmp_path, X=X, y=y)
        return {"temp_path": tmp_path, "n_samples": int(len(X))}
    except Exception as e:
        raise HTTPException(500, f"Failed to save temporary file: {str(e)}")

@app.post("/train")
async def train(req: TrainRequest, temp_path: str):
    # Validate temp_path exists and is a file
    if not os.path.isfile(temp_path):
        raise HTTPException(400, "Invalid or missing temp_path")

    # Validate it's in the temp directory for security
    if not temp_path.startswith(tempfile.gettempdir()):
        raise HTTPException(400, "Invalid temp_path: must be in temp directory")

    try:
        d = np.load(temp_path)
        X, y = d["X"], d["y"]
    except Exception as e:
        raise HTTPException(400, f"Failed to load data from temp file: {str(e)}")
    finally:
        # Clean up temp file after loading
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass  # Don't fail if cleanup fails

    try:
        model, norm_stats, (val_mse, test_mse) = train_model(
            X, y,
            batch_size=req.batch_size,
            max_epochs=req.max_epochs,
            lr=req.lr,
            weight_decay=req.weight_decay,
            patience=req.patience,
        )
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")

    # Store in state
    STATE.model = model
    STATE.norm_stats = norm_stats
    STATE.last_metrics = {
        "val_mse": float(val_mse),
        "test_mse": float(test_mse),
    }

    # persist artifacts
    try:
        os.makedirs("artifacts", exist_ok=True)
        save_model(model, norm_stats, "artifacts/latest.pkl")
    except Exception as e:
        # Don't fail training if artifact save fails
        pass

    return {"metrics": STATE.last_metrics}

class PredictRequest(BaseModel):
    X: list[conlist(float, min_length=5, max_length=5)]

@app.post("/predict")
def predict(req: PredictRequest):
    if STATE.model is None or STATE.norm_stats is None:
        raise HTTPException(400, "No model trained yet")

    if not req.X:
        raise HTTPException(400, "X cannot be empty")

    try:
        X = np.asarray(req.X, dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"Failed to convert X to array: {str(e)}")

    # Check for NaN or Inf values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise HTTPException(400, "X contains NaN or Inf values")

    # Validate shape
    if X.shape[1] != 5:
        raise HTTPException(400, f"Expected 5 features, got {X.shape[1]}")

    try:
        y = interpolate(STATE.model, STATE.norm_stats, X)
        return {"y": y.tolist()}
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")
