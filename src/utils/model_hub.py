import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


class Models_hub:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        self.models = {}

    def _get_scores(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return None

    def _slice_rows(self, X, indices):
        if hasattr(X, "iloc"):
            return X.iloc[indices]
        return X[indices]

    def add_loaded_model_to_leaderboard(self, model_name, loaded_model, X_valid, y_valid):
        pred_start = time.perf_counter()
        y_pred = loaded_model.predict(X_valid)
        predict_time_sec = time.perf_counter() - pred_start

        y_score = self._get_scores(loaded_model, X_valid)
        row = {
            "model": model_name,
            "fit_time_sec": 0.0,
            "predict_time_sec": predict_time_sec,
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred, zero_division=0),
            "recall": recall_score(y_valid, y_pred, zero_division=0),
            "f1": f1_score(y_valid, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_valid, y_score),
            "params": loaded_model.get_params() if hasattr(loaded_model, "get_params") else None,
        }
        self.results.append(row)
        self.models[model_name] = loaded_model
        return row, loaded_model

    def choice_best(self, model_class, X_train, y_train, model_name, params, general_params=None, n_splits=5):
        best_params = None
        best_auc = -np.inf

        for i in tqdm(range(len(params)), total=len(params), desc=f"{model_name} tuning"):
            params_to_use = {**params[i], **general_params} if general_params else params[i]
            try:
                row = self.fit_predict(
                    model_class,
                    X_train,
                    y_train,
                    model_name=model_name,
                    params=params_to_use,
                    n_splits=n_splits,
                    return_model=False,
                )

                if row["roc_auc"] > best_auc:
                    best_params = params_to_use
                    best_auc = row["roc_auc"]
            except Exception as e:
                print(f"Error with params {params_to_use}: {e}")

        best_model = model_class(**best_params)
        best_model.fit(X_train, y_train)
        self.models[model_name] = best_model

        return best_model, best_params

    def fit_predict(self, model_class, X, y, model_name, params=None, n_splits=5, return_model=True, fit_params=None):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
        }
        fit_times = []
        pred_times = []

        for train_idx, valid_idx in skf.split(X, y):
            X_train_fold, X_valid_fold = self._slice_rows(X, train_idx), self._slice_rows(X, valid_idx)
            y_train_fold, y_valid_fold = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx], y.iloc[valid_idx] if hasattr(y, "iloc") else y[valid_idx]

            model = model_class(**params) if params else model_class()

            fit_start = time.perf_counter()
            if fit_params:
                model.fit(X_train_fold, y_train_fold, **fit_params)
            else:
                model.fit(X_train_fold, y_train_fold)
            fit_times.append(time.perf_counter() - fit_start)

            pred_start = time.perf_counter()
            y_pred = model.predict(X_valid_fold)
            pred_times.append(time.perf_counter() - pred_start)

            y_score = self._get_scores(model, X_valid_fold)
            metrics["accuracy"].append(accuracy_score(y_valid_fold, y_pred))
            metrics["precision"].append(precision_score(y_valid_fold, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_valid_fold, y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_valid_fold, y_pred, zero_division=0))
            metrics["roc_auc"].append(roc_auc_score(y_valid_fold, y_score))

        row = {
            "model": model_name,
            "fit_time_sec": np.mean(fit_times),
            "predict_time_sec": np.mean(pred_times),
            "accuracy": np.mean(metrics["accuracy"]),
            "precision": np.mean(metrics["precision"]),
            "recall": np.mean(metrics["recall"]),
            "f1": np.mean(metrics["f1"]),
            "roc_auc": np.mean(metrics["roc_auc"]),
            "params": params,
        }
        self.results.append(row)

        if return_model:
            final_model = model_class(**params) if params else model_class()
            if fit_params:
                final_model.fit(X, y, **fit_params)
            else:
                final_model.fit(X, y)
            return row, final_model

        return row

    def leaderboard(self, only_best=False):
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        if only_best:
            def _best_index(g):
                if g["roc_auc"].notna().any():
                    return g["roc_auc"].idxmax()
                return g.index[0]

            best_indices = df.groupby("model", group_keys=False).apply(_best_index).values
            df = df.loc[best_indices].reset_index(drop=True)

        df_sorted = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

        return df_sorted.style.format(
            {
                "fit_time_sec": "{:.4f}",
                "predict_time_sec": "{:.4f}",
                "accuracy": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1": "{:.4f}",
                "roc_auc": "{:.4f}",
            }
        )
