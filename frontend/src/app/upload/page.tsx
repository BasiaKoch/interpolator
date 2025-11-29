"use client";
import { useState } from "react";
import { apiUpload } from "@/lib/api";

export default function UploadPage() {
  const [info, setInfo] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validation: Check file extension
    if (!file.name.endsWith('.pkl')) {
      setErr('Invalid file type. Please upload a .pkl file.');
      setSelectedFile(null);
      setInfo(null);
      return;
    }

    // Validation: Check file size (max 50MB)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      setErr('File too large. Maximum size is 50MB.');
      setSelectedFile(null);
      setInfo(null);
      return;
    }

    setSelectedFile(file);
    setErr(null);
    setInfo(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setErr(null);
    setInfo(null);

    try {
      const result = await apiUpload(selectedFile);
      setInfo(result);
      // Save temp_path to localStorage for auto-fill in train page
      if (result.temp_path) {
        localStorage.setItem('dataset_temp_path', result.temp_path);
      }
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Upload Dataset</h1>
      <p className="page-description">
        Upload a .pkl file containing 5-dimensional features (X) and target values (y).
        The file should contain a tuple (X, y) or dictionary with 'X' and 'y' keys.
      </p>

      <div className="form-group">
        <label htmlFor="file-upload">Select .pkl file</label>
        <input
          id="file-upload"
          type="file"
          accept=".pkl"
          onChange={handleFileSelect}
        />
        {selectedFile && (
          <div className="info-container" style={{ marginTop: '1rem' }}>
            <p><strong>Selected file:</strong> {selectedFile.name}</p>
            <p><strong>Size:</strong> {(selectedFile.size / 1024).toFixed(2)} KB</p>
          </div>
        )}
      </div>

      <button
        onClick={handleUpload}
        disabled={!selectedFile || loading}
      >
        {loading ? (
          <>
            <span className="loading"></span> Uploading...
          </>
        ) : (
          'Upload Dataset'
        )}
      </button>

      {err && (
        <div className="error-container">
          <div className="error-title">❌ Error</div>
          <p>{err}</p>
        </div>
      )}

      {info && (
        <div className="result-container">
          <div className="result-title">✅ Upload Successful</div>
          <div className="card">
            <p><strong>Number of samples:</strong> {info.n_samples}</p>
            <p><strong>Temporary path:</strong> <code>{info.temp_path}</code></p>
            <p style={{ marginTop: '1rem', color: 'var(--secondary-color)', fontWeight: 600 }}>
              ✓ Dataset ready! The path has been automatically saved for training.
            </p>
          </div>
          <div style={{ marginTop: '1rem' }}>
            <a href="/train" className="button button-secondary">
              Go to Training →
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

