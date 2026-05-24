import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from sklearn.metrics import mean_squared_error, r2_score 
import warnings 
warnings.filterwarnings('ignore') 
 
class SimpleLSTMPredictor(nn.Module): 
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1): 
        super().__init__() 
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2) 
        self.fc = nn.Linear(hidden_dim, output_dim) 
    def forward(self, x): 
        lstm_out, _ = self.lstm(x) 
        return self.fc(lstm_out[:, -1, :]) 
 
def run_prediction(fused_state=None, data_dir="data/raw"): 
    print("=" * 60) 
    print("PHASE 2: GNN-LSTM PREDICTIVE DIGITAL TWIN") 
    print("=" * 60) 
    sensors = pd.read_csv(f"{data_dir}/synthetic_dataset_03_sensor_network.csv") 
    spatial = pd.read_csv(f"{data_dir}/synthetic_dataset_01_spatial_grid.csv") 
    seq_len = 6 
    cell_ph_series = {} 
    for cell_id in spatial['cell_id'].values[:200]: 
        cell_data = sensors[sensors['cell_id'] == cell_id].sort_values('timestep') 
        if len(cell_data)  + 1: 
            ph_values = [] 
            for t in range(24): 
                t_data = cell_data[cell_data['timestep'] == t] 
                if len(t_data) 
                    ph_values.append(t_data['soil_pH'].values[0]) 
                else: 
                    ph_values.append(np.nan) 
                cell_ph_series[cell_id] = ph_values 
    X_seq, y_seq = [], [] 
    for vals in cell_ph_series.values(): 
        vals_series = pd.Series(vals).interpolate().fillna(6.5) 
        vals = vals_series.values 
        for t in range(seq_len, 24): 
            X_seq.append(vals[t-seq_len:t]) 
            y_seq.append(vals[t]) 
    X_seq = np.array(X_seq).reshape(-1, seq_len, 1) 
    y_seq = np.array(y_seq) 
    model = SimpleLSTMPredictor() 
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    X_tensor = torch.FloatTensor(X_seq[:50]) 
    y_tensor = torch.FloatTensor(y_seq[:50]).reshape(-1, 1) 
    for epoch in range(100): 
        optimizer.zero_grad() 
        output = model(X_tensor) 
        loss = criterion(output, y_tensor) 
        loss.backward() 
        optimizer.step() 
    torch.save(model.state_dict(), 'gnn_lstm_model.pth') 
    print("Model saved to gnn_lstm_model.pth") 
    print("PHASE 2 COMPLETE") 
    return model 
 
if __name__ == "__main__": 
    run_prediction() 
