import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
    import lightgbm as lgb
    
except ImportError as e:
    print(f"HATA: kütüphane eksik: {e.name}")
    sys.exit()


def get_fingerprint(smiles_str, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        return list(fp)
    except:
        return None

def load_data(filename, task_name):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"   HATA: {filename} bulunamadı.")
        return None, None

    if 'task' not in df.columns:
         df_task = df.copy()
    else:
         df_task = df[df['task'] == task_name].copy()

    rename_map = {'prompt': 'smiles', 'target': 'value', 'SMILES': 'smiles', 'Target': 'value'}
    df_task = df_task.rename(columns=rename_map)

    df_task['value'] = pd.to_numeric(df_task['value'], errors='coerce')
    df_task = df_task.dropna(subset=['smiles', 'value'])
    
    if len(df_task) == 0:
        return None, None

    df_task['fingerprint'] = df_task['smiles'].apply(get_fingerprint)
    df_task = df_task.dropna(subset=['fingerprint'])

    time.sleep(11)
    X = np.array(df_task['fingerprint'].tolist())
    y = df_task['value'].values
    return X, y


tasks = ['band gap bulk', 'band gap chain', 'bandgap-crystal']

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}



for task in tasks:
    print(f"\n{'#'*60}")
    print(f" ÖZELLİK: {task}")
    print(f"{'#'*60}")

    X_train, y_train = load_data('./Bandgap Bulk/dataset3.csv', task)
    X_test, y_test = load_data('./Bandgap Bulk/dataset4.csv', task)

    if X_train is None or X_test is None:
        continue

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    best_r2 = -np.inf
    best_model_name = ""
    best_y_pred = []
    best_rmse = 0

    print(f"\n   {'Model':<15} | {'R2':<8} | {'RMSE':<8}")
    print(f"   {'-'*35}")
    
    for name, model in models.items():
        if name in ["RandomForest", "XGBoost", "LightGBM"]:
            X_tr, X_te = X_train, X_test
        else:
            X_tr, X_te = X_train_scaled, X_test_scaled
        
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   {name:<15} | {r2:<8.4f} | {rmse:<8.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_y_pred = y_pred
            best_rmse = rmse

    print(f"\n   >>>  {best_model_name} (R2: {best_r2:.4f}, RMSE: {best_rmse:.4f})")

    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_test, best_y_pred, alpha=0.6, color='blue', edgecolors='k', label='Veri Noktaları')

    min_val = min(y_test.min(), best_y_pred.min())
    max_val = max(y_test.max(), best_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='İdeal (y=x)')
    
    plt.title(f"En İyi Model Performansı: {best_model_name}\n$R^2 = {best_r2:.3f}$ ", fontsize=24)
    plt.xlabel(f"Gerçek {task} Değerleri ", fontsize=20)
    plt.ylabel(f"Tahmin Edilen {task} Değerleri", fontsize=20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename_safe = task.replace(" ", "_")
    plt.savefig(f"best_model_{filename_safe}.png")
    plt.show()
    print(f"   Grafik 'best_model_{filename_safe}.png' olarak kaydedildi.\n")

