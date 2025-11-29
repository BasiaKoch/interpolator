"use client";
import { useState, useEffect } from "react";
import { apiTrain } from "@/lib/api";

export default function TrainPage() {
  const [tempPath, setTempPath] = useState("");
  const [hidden, setHidden] = useState("128,64,32");
  const [lr, setLr] = useState("0.001");
  const [maxEpochs, setMaxEpochs] = useState("150");
  const [batchSize, setBatchSize] = useState("256");
  const [patience, setPatience] = useState("15");
  const [scaleY, setScaleY] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Auto-fill temp_path from localStorage on component mount
  useEffect(() => {
    const savedTempPath = localStorage.getItem('dataset_temp_path');
    if (savedTempPath) {
      setTempPath(savedTempPath);
    }
  }, []);

  const handleTrain = async () => {
    // Validation
    if (!tempPath.trim()) {
      setErr("Please enter the temp_path from the upload step");
      return;
    }

    const hiddenArray = hidden.split(",").map(x => parseInt(x.trim()));
    if (hiddenArray.some(isNaN)) {
      setErr("Invalid hidden layer sizes. Use comma-separated numbers (e.g., 64,32,16)");
      return;
    }

    const lrNum = parseFloat(lr);
    if (isNaN(lrNum) || lrNum <= 0) {
      setErr("Learning rate must be a positive number");
      return;
    }

    const epochsNum = parseInt(maxEpochs);
    if (isNaN(epochsNum) || epochsNum <= 0) {
      setErr("Max epochs must be a positive integer");
      return;
    }

    const batchNum = parseInt(batchSize);
    if (isNaN(batchNum) || batchNum <= 0) {
      setErr("Batch size must be a positive integer");
      return;
    }

    const patienceNum = parseInt(patience);
    if (isNaN(patienceNum) || patienceNum <= 0) {
      setErr("Patience must be a positive integer");
      return;
    }

    setLoading(true);
    setErr(null);
    setMetrics(null);

    try {
      const payload = {
        hidden: hiddenArray,
        lr: lrNum,
        max_epochs: epochsNum,
        batch_size: batchNum,
        patience: patienceNum,
        scale_y: scaleY
      };
      const result = await apiTrain(tempPath, payload);
      setMetrics(result.metrics);
      // Clear temp_path from localStorage after successful training
      localStorage.removeItem('dataset_temp_path');
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Train Model</h1>
      <p className="page-description">
        Configure the neural network hyperparameters and start training on your uploaded dataset.
      </p>

      <div className="form-group">
        <label htmlFor="temp-path">
          Temporary Path <span style={{ color: 'var(--error-color)' }}>*</span>
        </label>
        <input
          id="temp-path"
          type="text"
          value={tempPath}
          onChange={e => setTempPath(e.target.value)}
          placeholder="Auto-filled from upload step"
        />
        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
          {tempPath ? '✓ Path loaded automatically from upload' : 'Upload a dataset first to auto-fill this field'}
        </p>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: '1rem' }}>Hyperparameters</h3>

        <div className="form-group">
          <label htmlFor="hidden">Hidden Layer Sizes</label>
          <input
            id="hidden"
            type="text"
            value={hidden}
            onChange={e => setHidden(e.target.value)}
            placeholder="e.g., 64,32,16"
          />
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
            Comma-separated layer sizes (e.g., 64,32,16 creates 3 hidden layers)
          </p>
        </div>

        <div className="input-grid">
          <div className="form-group">
            <label htmlFor="lr">Learning Rate</label>
            <input
              id="lr"
              type="number"
              step="0.0001"
              value={lr}
              onChange={e => setLr(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="max-epochs">Max Epochs</label>
            <input
              id="max-epochs"
              type="number"
              value={maxEpochs}
              onChange={e => setMaxEpochs(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="batch-size">Batch Size</label>
            <input
              id="batch-size"
              type="number"
              value={batchSize}
              onChange={e => setBatchSize(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="patience">Early Stop Patience</label>
            <input
              id="patience"
              type="number"
              value={patience}
              onChange={e => setPatience(e.target.value)}
            />
          </div>
        </div>

        <div className="form-group">
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={scaleY}
              onChange={e => setScaleY(e.target.checked)}
              style={{ width: 'auto' }}
            />
            <span>Scale target values (y)</span>
          </label>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem', marginLeft: '1.75rem' }}>
            Enable to standardize target values before training
          </p>
        </div>
      </div>

      <button onClick={handleTrain} disabled={loading}>
        {loading ? (
          <>
            <span className="loading"></span> Training...
          </>
        ) : (
          '🚀 Start Training'
        )}
      </button>

      {err && (
        <div className="error-container">
          <div className="error-title">❌ Error</div>
          <p>{err}</p>
        </div>
      )}

      {metrics && (
        <div className="result-container">
          <div className="result-title">✅ Training Complete</div>
          <div className="card">
            <h3>Validation Metrics</h3>
            <div className="input-grid" style={{ marginTop: '1rem' }}>
              <div>
                <p style={{ color: 'var(--text-secondary)' }}>Mean Squared Error</p>
                <p className="result-value" style={{ fontSize: '1.5rem' }}>
                  {metrics.val_mse?.toFixed(6) || 'N/A'}
                </p>
              </div>
              <div>
                <p style={{ color: 'var(--text-secondary)' }}>R² Score</p>
                <p className="result-value" style={{ fontSize: '1.5rem' }}>
                  {metrics.val_r2?.toFixed(6) || 'N/A'}
                </p>
              </div>
            </div>
            <details style={{ marginTop: '1.5rem' }}>
              <summary style={{ cursor: 'pointer', fontWeight: 600 }}>View Raw Metrics</summary>
              <pre style={{ marginTop: '0.5rem' }}>{JSON.stringify(metrics, null, 2)}</pre>
            </details>
          </div>
          <div style={{ marginTop: '1rem' }}>
            <a href="/predict" className="button button-secondary">
              Make Predictions →
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

