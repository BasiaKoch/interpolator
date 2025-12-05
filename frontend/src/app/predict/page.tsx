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

  const MIN_VALUE = -5;
  const MAX_VALUE = 5;
  const STEP = 0.1;

  const handleSliderChange = (index: number, value: string) => {
    const newX = [...x];
    const numValue = parseFloat(value);
    newX[index] = isNaN(numValue) ? 0 : numValue;
    setX(newX);
  };

  const handleRandomize = () => {
    const randomX = Array(5).fill(0).map(() =>
      parseFloat((Math.random() * (MAX_VALUE - MIN_VALUE) + MIN_VALUE).toFixed(2))
    );
    setX(randomX);
    setY(null);
    setErr(null);
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
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.875rem' }}>
          Adjust the sliders to set values for all 5 features (range: {MIN_VALUE} to {MAX_VALUE}):
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginBottom: '2rem' }}>
          {x.map((value, index) => (
            <div key={index} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label htmlFor={`slider-${index}`} style={{ fontWeight: 500 }}>
                  {featureLabels[index]}
                </label>
                <span style={{
                  backgroundColor: 'var(--primary-color)',
                  color: 'white',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '4px',
                  fontWeight: 600,
                  minWidth: '60px',
                  textAlign: 'center',
                  fontSize: '0.875rem'
                }}>
                  {value.toFixed(2)}
                </span>
              </div>
              <input
                id={`slider-${index}`}
                type="range"
                min={MIN_VALUE}
                max={MAX_VALUE}
                step={STEP}
                value={value}
                onChange={e => handleSliderChange(index, e.target.value)}
                style={{
                  width: '100%',
                  height: '8px',
                  borderRadius: '5px',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                <span>{MIN_VALUE}</span>
                <span>{MAX_VALUE}</span>
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
          <button
            onClick={handleRandomize}
            disabled={loading}
            style={{
              flex: 1,
              backgroundColor: 'var(--text-secondary)',
              padding: '0.75rem'
            }}
          >
            🎲 Randomize
          </button>
          <button
            onClick={handleReset}
            disabled={loading}
            style={{
              flex: 1,
              backgroundColor: 'var(--text-secondary)',
              padding: '0.75rem'
            }}
          >
            ↺ Reset
          </button>
        </div>

        <button
          onClick={handlePredict}
          disabled={loading}
          style={{
            width: '100%',
            padding: '1rem',
            fontSize: '1.125rem',
            fontWeight: 600,
            backgroundColor: 'var(--primary-color)',
            border: 'none',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1,
            transition: 'all 0.2s'
          }}
        >
          {loading ? (
            <>
              <span className="loading"></span> Predicting...
            </>
          ) : (
            '🎯 Make Prediction'
          )}
        </button>
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
