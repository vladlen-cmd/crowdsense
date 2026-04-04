import numpy as np
import time
import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class CrowdPredictor:
    FEATURE_DIM = 5
    HIDDEN_DIM = 32
    OUTPUT_DIM = 12

    def __init__(
        self,
        zone_name: str,
        max_capacity: int,
        model_path: Optional[str] = None,
        sequence_len: int = 24,
        prediction_interval_min: int = 5,
    ):
        self.zone_name = zone_name
        self.max_capacity = max_capacity
        self.sequence_len = sequence_len
        self.prediction_interval_min = prediction_interval_min

        self._buffer: deque = deque(maxlen=sequence_len)

        self._weights = None
        if model_path and Path(model_path).exists():
            self._load_weights(model_path)
        else:
            logger.warning(
                f'No model weights found at {model_path}. '
                'Using heuristic fallback until trained weights are available.'
            )

        self._last_prediction: list[float] = []
        self._last_prediction_time: float = 0

    @staticmethod
    def _time_features(ts: float = None) -> tuple[float, float, float, float]:
        import math
        if ts is None:
            ts = time.time()
        import datetime
        dt = datetime.datetime.fromtimestamp(ts)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin  = math.sin(2 * math.pi * dow / 7)
        dow_cos  = math.cos(2 * math.pi * dow / 7)

        return hour_sin, hour_cos, dow_sin, dow_cos

    def push_observation(self, count: int, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()

        norm_count = min(count / self.max_capacity, 1.0)
        h_sin, h_cos, d_sin, d_cos = self._time_features(timestamp)
        feature_vec = np.array([norm_count, h_sin, h_cos, d_sin, d_cos], dtype=np.float32)
        self._buffer.append(feature_vec)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -50, 50))

    def _lstm_cell(self, x, h_prev, c_prev, weights: dict) -> tuple:
        xh = np.concatenate([x, h_prev])

        gates = xh @ weights['W_gates'] + weights['b_gates']
        i_size = self.HIDDEN_DIM

        i_gate = self._sigmoid(gates[0 * i_size: 1 * i_size])
        f_gate = self._sigmoid(gates[1 * i_size: 2 * i_size])
        g_gate = self._tanh(gates[2 * i_size: 3 * i_size])
        o_gate = self._sigmoid(gates[3 * i_size: 4 * i_size])

        c = f_gate * c_prev + i_gate * g_gate
        h = o_gate * self._tanh(c)
        return h, c

    def _run_lstm(self, sequence: np.ndarray) -> np.ndarray:
        if self._weights is None:
            return self._heuristic_forecast(sequence)

        w = self._weights
        h = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        c = np.zeros(self.HIDDEN_DIM, dtype=np.float32)

        for t in range(len(sequence)):
            h, c = self._lstm_cell(sequence[t], h, c, w)

        output = h @ w['W_out'] + w['b_out']
        return np.clip(output, 0.0, 1.0)

    def _heuristic_forecast(self, sequence: np.ndarray) -> np.ndarray:
        if len(sequence) == 0:
            return np.zeros(self.OUTPUT_DIM)

        recent = sequence[-min(6, len(sequence)):, 0]
        mean_level = float(recent.mean())
        trend = float(recent[-1] - recent[0]) / max(len(recent) - 1, 1) if len(recent) > 1 else 0.0

        forecast = []
        for step in range(self.OUTPUT_DIM):
            val = mean_level + trend * step * 0.5
            val = max(0.0, min(1.0, val))
            forecast.append(val)

        return np.array(forecast, dtype=np.float32)

    def predict(self, force: bool = False) -> list[dict]:
        cache_ttl = self.prediction_interval_min * 30
        if not force and (time.time() - self._last_prediction_time) < cache_ttl:
            return self._last_prediction

        if len(self._buffer) < 3:
            logger.debug('Not enough history for prediction yet.')
            return []

        sequence = np.array(list(self._buffer), dtype=np.float32)
        raw_forecast = self._run_lstm(sequence)

        results = []
        for step, norm_val in enumerate(raw_forecast):
            minutes_ahead = (step + 1) * self.prediction_interval_min
            predicted_count = int(norm_val * self.max_capacity)
            results.append({
                'minutes_ahead': minutes_ahead,
                'predicted_count': predicted_count,
                'occupancy_pct': round(norm_val * 100, 1),
            })

        self._last_prediction = results
        self._last_prediction_time = time.time()
        return results

    def peak_warning(self, warn_threshold: float = 0.80) -> dict | None:
        forecast = self.predict()
        for step in forecast:
            if step['occupancy_pct'] / 100 >= warn_threshold:
                return step
        return None

    def _load_weights(self, path: str):
        try:
            with open(path) as f:
                raw = json.load(f)
            self._weights = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
            logger.info(f'Loaded LSTM weights from {path}')
        except Exception as e:
            logger.error(f'Failed to load weights: {e}')
            self._weights = None

    def save_weights(self, path: str):
        if self._weights is None:
            logger.warning('No weights to save.')
            return
        with open(path, 'w') as f:
            json.dump({k: v.tolist() for k, v in self._weights.items()}, f)
        logger.info(f'Weights saved to {path}')

def train_model(
    training_csv: str,
    zone_name: str,
    max_capacity: int,
    output_weights_path: str,
    epochs: int = 100,
    lr: float = 0.01,
):
    try:
        import torch
        import torch.nn as nn
        import pandas as pd
    except ImportError:
        print('Training requires: pip install torch pandas')
        return

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))

    df = pd.read_csv(training_csv, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    sequences, targets = [], []
    seq_len = 24
    pred_steps = 12

    for i in range(seq_len, len(df) - pred_steps):
        window = df.iloc[i - seq_len: i]
        target = df.iloc[i: i + pred_steps]['count'].values / max_capacity

        feats = []
        for _, row in window.iterrows():
            ts = row['timestamp'].timestamp()
            p = CrowdPredictor(zone_name, max_capacity)
            norm_count = row['count'] / max_capacity
            h_sin, h_cos, d_sin, d_cos = p._time_features(ts)
            feats.append([norm_count, h_sin, h_cos, d_sin, d_cos])

        sequences.append(feats)
        targets.append(target.tolist())

    X = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    model = LSTMModel(5, 32, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}')

    state = model.state_dict()
    W_ih = state['lstm.weight_ih_l0'].numpy()
    W_hh = state['lstm.weight_hh_l0'].numpy()
    b_ih = state['lstm.bias_ih_l0'].numpy()
    b_hh = state['lstm.bias_hh_l0'].numpy()

    W_gates = np.vstack([W_ih.T, W_hh.T])
    b_gates = b_ih + b_hh

    weights = {
        'W_gates': W_gates.tolist(),
        'b_gates': b_gates.tolist(),
        'W_out': state['fc.weight'].numpy().T.tolist(),
        'b_out': state['fc.bias'].numpy().tolist(),
    }

    with open(output_weights_path, 'w') as f:
        json.dump(weights, f)

    print(f'\nWeights saved → {output_weights_path}')
    print('Copy this file to your Raspberry Pi.')
