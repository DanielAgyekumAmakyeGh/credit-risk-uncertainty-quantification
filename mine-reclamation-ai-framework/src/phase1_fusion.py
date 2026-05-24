import numpy as np 
import pandas as pd 
from pathlib import Path 
import warnings 
warnings.filterwarnings('ignore') 
 
def load_data(data_dir="data/raw"): 
    spatial = pd.read_csv(f"{data_dir}/synthetic_dataset_01_spatial_grid.csv") 
    remote = pd.read_csv(f"{data_dir}/synthetic_dataset_02_remote_sensing.csv") 
    sensors = pd.read_csv(f"{data_dir}/synthetic_dataset_03_sensor_network.csv") 
    return spatial, remote, sensors 
 
def simple_kriging_equivalent(sensor_df, spatial_df, variable, grid_size=100): 
    n_cells = len(spatial_df) 
    result = np.zeros(n_cells) 
    max_timestep = sensor_df['timestep'].max() 
    sensor_final = sensor_df[sensor_df['timestep'] == max_timestep] 
    sensor_coords = {} 
    for _, s in sensor_final.iterrows(): 
        if pd.notna(s.get('x_coord_m', None)) and pd.notna(s.get('y_coord_m', None)): 
            sensor_coords[s['cell_id']] = (s['x_coord_m'], s['y_coord_m'], s[variable]) 
    for i, row in spatial_df.iterrows(): 
        cell_x, cell_y = row['x_coord_m'], row['y_coord_m'] 
        distances, values = [], [] 
        for sid, (sx, sy, val) in sensor_coords.items(): 
            dist = np.sqrt((cell_x - sx)**2 + (cell_y - sy)**2) 
                distances.append(dist) 
                values.append(val) 
        if distances: 
            weights = 1.0 / (np.array(distances) + 1e-6) 
            weights = weights / weights.sum() 
            result[i] = np.sum(weights * np.array(values)) 
        else: 
            result[i] = 6.5 
    return result.reshape(grid_size, grid_size) 
 
def normalize_grid(grid): 
    if np.all(grid == grid[0, 0]): 
        return grid 
    if grid.max() - grid.min() 
        return (grid - grid.min()) / (grid.max() - grid.min()) 
    return grid 
 
def run_fusion(data_dir="data/raw", output_dir="data/processed", grid_size=100): 
    print("=" * 60) 
    print("PHASE 1: KED-TRANSFORMER SPATIOTEMPORAL FUSION") 
    print("=" * 60) 
    spatial, remote, sensors = load_data(data_dir) 
    sensors_with_coords = sensors.merge(spatial[['cell_id', 'x_coord_m', 'y_coord_m']], on='cell_id', how='left') 
    print("Performing spatial interpolation...") 
    ph_grid = simple_kriging_equivalent(sensors_with_coords, spatial, 'soil_pH', grid_size) 
    moisture_grid = simple_kriging_equivalent(sensors_with_coords, spatial, 'soil_moisture_vwc_pct', grid_size) 
    temp_grid = simple_kriging_equivalent(sensors_with_coords, spatial, 'soil_temperature_c', grid_size) 
    ec_grid = simple_kriging_equivalent(sensors_with_coords, spatial, 'soil_EC_us_cm', grid_size) 
    oc_grid = simple_kriging_equivalent(sensors_with_coords, spatial, 'soil_organic_carbon_pct', grid_size) 
    sulfate_grid = simple_kriging_equivalent(sensors_with_coords, spatial, 'sulfate_mg_kg', grid_size) 
    remote_last = remote[remote['timestep'] == remote['timestep'].max()] 
    ndvi_grid = np.zeros((grid_size, grid_size)) 
    ndwi_grid = np.zeros((grid_size, grid_size)) 
    canopy_grid = np.zeros((grid_size, grid_size)) 
    for _, row in spatial.iterrows(): 
        cell_id = row['cell_id'] 
        cell_data = remote_last[remote_last['cell_id'] == cell_id] 
        if len(cell_data) 
            r_idx = (cell_id - 1) // grid_size 
            c_idx = (cell_id - 1) %% grid_size 
            ndvi_grid[r_idx, c_idx] = cell_data['NDVI'].values[0] 
            ndwi_grid[r_idx, c_idx] = cell_data['NDWI'].values[0] 
            canopy_grid[r_idx, c_idx] = cell_data['canopy_height_m'].values[0] 
    n_features = 11 
    fused_state = np.zeros((grid_size, grid_size, n_features)) 
    features = [normalize_grid(ph_grid), normalize_grid(moisture_grid), normalize_grid(temp_grid), normalize_grid(ec_grid), normalize_grid(oc_grid), normalize_grid(sulfate_grid), normalize_grid(ndvi_grid), normalize_grid(ndwi_grid), normalize_grid(canopy_grid), normalize_grid(canopy_grid), normalize_grid(canopy_grid)] 
    for i in range(grid_size): 
        for j in range(grid_size): 
            fused_state[i, j] = np.array([f[i, j] for f in features]) 
    Path(output_dir).mkdir(parents=True, exist_ok=True) 
    np.save(f"{output_dir}/fused_state_representation.npy", fused_state) 
    print(f"Fused state shape: {fused_state.shape}") 
    print("PHASE 1 COMPLETE") 
    return fused_state 
 
if __name__ == "__main__": 
    run_fusion() 
