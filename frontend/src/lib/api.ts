/**
 * API client for communicating with the FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Upload a .pkl dataset file
 */
export async function apiUpload(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
  }

  return await response.json();
}

/**
 * Train the model with specified hyperparameters
 */
export async function apiTrain(tempPath: string, config: {
  hidden: number[];
  lr: number;
  max_epochs: number;
  batch_size: number;
  patience: number;
  scale_y: boolean;
}) {
  const response = await fetch(`${API_BASE_URL}/train/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      temp_path: tempPath,
      ...config,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Training failed' }));
    throw new Error(errorData.detail || `Training failed with status ${response.status}`);
  }

  return await response.json();
}

/**
 * Make predictions using the trained model
 */
export async function apiPredict(X: number[][]) {
  const response = await fetch(`${API_BASE_URL}/predict/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ X }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(errorData.detail || `Prediction failed with status ${response.status}`);
  }

  return await response.json();
}
