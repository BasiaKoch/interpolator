from ..data.loader import split_and_standardize
from ..models.mlp import MLPRegressor, MLPConfig
from sklearn.metrics import mean_squared_error, r2_score

def train_from_arrays(X, y, cfg: MLPConfig = MLPConfig(), scale_y=False):
    ds = split_and_standardize(X, y, scale_y=scale_y)
    model = MLPRegressor(cfg).fit(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    preds = model.predict(ds.X_val)
    metrics = {"val_mse": float(mean_squared_error(ds.y_val, preds)),
               "val_r2":  float(r2_score(ds.y_val, preds))}
    return model, ds, metrics

