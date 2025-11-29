"use client";
import { useState } from "react";
import { apiPredict } from "@/lib/api";

export default function PredictPage() {
  const [x, setX] = useState<number[]>([0, 0, 0, 0, 0]);
  const [y, setY] = useState<number[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const featureLabels = [
    "Feature 1",
    "Feature 2",
    "Feature 3",
    "Feature 4",
    "Feature 5"
  ];

  const handleInputChange = (index: number, value: string) => {
    const newX = [...x];
    const numValue = parseFloat(value);
    newX[index] = isNaN(numValue) ? 0 : numValue;
    setX(newX);
  };

  const handlePredict = async () => {
    // Validation
    if (x.some(v => isNaN(v))) {
      setErr("All feature values must be valid numbers");
      return;
    }

    if (x.some(v => !isFinite(v))) {
      setErr("Feature values cannot be infinite");
      return;
    }

    setLoading(true);
    setErr(null);
    setY(null);

    try {
      const result = await apiPredict([x]);
      setY(result.y);
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setX([0, 0, 0, 0, 0]);
    setY(null);
    setErr(null);
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Make Predictions</h1>
      <p className="page-description">
        Enter 5-dimensional feature values to get a prediction from the trained model.
        Make sure you have trained a model first.
      </p>

      <div className="card">
        <h3 style={{ marginBottom: '1rem' }}>Input Features</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.875rem' }}>
          Enter values for all 5 features:
        </p>

        <div className="input-grid">
          {x.map((value, index) => (
            <div key={index} className="form-group">
              <label htmlFor={`feature-${index}`}>
                {featureLabels[index]}
              </label>
              <input
                id={`feature-${index}`}
                type="number"
                step="any"
                value={value}
                onChange={e => handleInputChange(index, e.target.value)}
                placeholder={`Enter ${featureLabels[index].toLowerCase()}`}
              />
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
          <button onClick={handlePredict} disabled={loading}>
            {loading ? (
              <>
                <span className="loading"></span> Predicting...
              </>
            ) : (
              '🎯 Predict'
            )}
          </button>
          <button
            onClick={handleReset}
            disabled={loading}
            style={{ backgroundColor: 'var(--text-secondary)' }}
          >
            Reset
          </button>
        </div>
      </div>

      {err && (
        <div className="error-container">
          <div className="error-title">❌ Error</div>
          <p>{err}</p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
            Make sure you have trained a model on the Train page first.
          </p>
        </div>
      )}

      {y !== null && (
        <div className="result-container">
          <div className="result-title">✅ Prediction Result</div>
          <div className="card">
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                Predicted Value (ŷ)
              </p>
              <div className="result-value">
                {y[0].toFixed(6)}
              </div>
            </div>

            <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-color)' }}>
              <p style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Input Summary:</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.5rem' }}>
                {x.map((value, index) => (
                  <div key={index} style={{ fontSize: '0.875rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{featureLabels[index]}:</span>{' '}
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div style={{ marginTop: '1rem' }}>
            <button onClick={handleReset} className="button-secondary">
              Make Another Prediction
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
