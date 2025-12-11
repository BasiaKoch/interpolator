import Link from 'next/link'

export default function Home() {
  return (
    <div className="home-container">
      <h1 className="home-title">5D Interpolator</h1>
      <p className="home-description">
        A machine learning system for training and predicting on 5-dimensional datasets using neural networks
      </p>

      <div className="home-cards">
        <Link href="/upload" className="home-card">
          <h2 className="home-card-title">📤 Upload Dataset</h2>
          <p className="home-card-description">
            Upload your .pkl file containing 5D features (X) and target values (y) for training
          </p>
        </Link>

        <Link href="/train" className="home-card">
          <h2 className="home-card-title">🧠 Train Model</h2>
          <p className="home-card-description">
            Configure hyperparameters and train a neural network on your uploaded dataset
          </p>
        </Link>

        <Link href="/predict" className="home-card">
          <h2 className="home-card-title">🎯 Make Predictions</h2>
          <p className="home-card-description">
            Use the trained model to make predictions on new 5-dimensional input data
          </p>
        </Link>
      </div>
    </div>
  );
}

