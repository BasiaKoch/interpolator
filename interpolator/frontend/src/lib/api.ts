/**
 * API client for communicating with the FastAPI backend
 */

// Default to Docker deployment port (8001). For local dev without Docker, set NEXT_PUBLIC_API_URL=http://localhost:8000
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

/**
 * Upload a .pkl dataset file
 */
export async function apiUpload(file: File) {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error: any) {
    // Better error handling
    if (error.message) {
      throw error;
    }
    if (error.name === 'TypeError' && error.message?.includes('fetch')) {
      throw new Error('Cannot connect to backend. Make sure the API server is running on ' + API_BASE_URL);
    }
    throw new Error('Upload failed: ' + (error.toString ? error.toString() : JSON.stringify(error)));
  }
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
  try {
    const response = await fetch(`${API_BASE_URL}/train?temp_path=${encodeURIComponent(tempPath)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Training failed' }));
      throw new Error(errorData.detail || `Training failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error: any) {
    // Better error handling
    if (error.message) {
      throw error;
    }
    if (error.name === 'TypeError' && error.message?.includes('fetch')) {
      throw new Error('Cannot connect to backend. Make sure the API server is running on ' + API_BASE_URL);
    }
    throw new Error('Training failed: ' + (error.toString ? error.toString() : JSON.stringify(error)));
  }
}

/**
 * Make predictions using the trained model
 */
export async function apiPredict(X: number[][]) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
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
  } catch (error: any) {
    // Better error handling
    if (error.message) {
      throw error;
    }
    if (error.name === 'TypeError' && error.message?.includes('fetch')) {
      throw new Error('Cannot connect to backend. Make sure the API server is running on ' + API_BASE_URL);
    }
    throw new Error('Prediction failed: ' + (error.toString ? error.toString() : JSON.stringify(error)));
  }
}
