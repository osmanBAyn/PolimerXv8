import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
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
    sys.exit()


def get_features(smiles_str, n_bits=2048):
    try:
        clean_smi = str(smiles_str).replace('*', '[H]')
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None: return None
        
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits))
        
        desc = [
            Descriptors.MolWt(mol),            # Ağırlık 
            Descriptors.MolLogP(mol),          # Hidrofobiklik
            Descriptors.TPSA(mol),             # Polarite
            Descriptors.MolMR(mol),            # Hacim/Kırılma
            Descriptors.NumRotatableBonds(mol), # Esneklik 
            Descriptors.HeavyAtomCount(mol),   # Atom sayısı
            Descriptors.FractionCSP3(mol)      # Zincir yapısı 
        ]
        return fp + desc
    except:
        return None

def load_data(filename, target_col, augment=True, augment_factor=1):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"   HATA: {filename} bulunamadı.")
        return None, None

    if target_col not in df.columns:
        print(f"   HATA: '{target_col}' sütunu bulunamadı.")
        return None, None

    smi_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    df = df.rename(columns={smi_col: 'smiles', target_col: 'value'})
    
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['smiles', 'value'])

    time.sleep(10)
    
    if augment:
        augmented_rows = []
        for index, row in df.iterrows():
            augmented_rows.append(row)
            
            for _ in range(augment_factor):
                new_row = row.copy()
                noise = np.random.normal(0, 0.05) * row['value']
                new_row['value'] = row['value'] + noise
                
                new_row['name'] = f"{row['name']}_aug"
                augmented_rows.append(new_row)
        
        df = pd.DataFrame(augmented_rows)

    df['features'] = df['smiles'].apply(get_features)
    
    df = df.dropna(subset=['features'])

    if len(df) == 0:
        return None, None

    X = np.array(df['features'].tolist())
    y = df['value'].values
    return X, y

dataset_path = './Isıl Genleşme/thermal_properties.csv'

targets = {
    'Thermal_Cond': 'Isıl İletkenlik',
    'CTE': 'Isıl Genleşme Katsayısı'
}

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1, n_jobs=-1, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0),
    "MLP": MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42)
}



for target_key, target_name in targets.items():
    print(f"\n{'='*60}")
    print(f" ÖZELLİK: {target_name} ({target_key})")
    print(f"{'='*60}")

    X, y = load_data(dataset_path, target_key)

    if X is None or y is None:
        print(f" {target_key} için veri bulunamadı.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    results_data = []
    best_r2 = -np.inf
    best_model_name = ""
    best_y_pred = []
    best_rmse = 0

    print(f"\n   {'Model':<15} | {'R2':<8} | {'RMSE':<8} | {'MAE':<8}")
    print(f"   {'-'*45}")
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   {name:<15} | {r2:<8.4f} | {rmse:<8.4f} | {mae:<8.4f}")
        
        results_data.append({
            "Model": name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_y_pred = y_pred
            best_rmse = rmse

    print(f"\n   >>> KAZANAN ({target_key}): {best_model_name} (R2: {best_r2:.4f})")
    
    df_res = pd.DataFrame(results_data).sort_values(by="R2", ascending=False)
    print("\n", df_res)

    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_test, best_y_pred, alpha=0.6, color='blue', edgecolor='k', label='Veri Noktaları')
    
    min_val = min(y_test.min(), best_y_pred.min())
    max_val = max(y_test.max(), best_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='İdeal (y=x)')
    
    plt.title(f"En İyi Model Performansı: {best_model_name}\n$R^2 = {best_r2:.3f}$ ", fontsize=24)
    plt.xlabel(f"Gerçek {target_key} değeri", fontsize=20)
    plt.ylabel(f"Tahmin Edilen {target_key} değeri", fontsize=20)
    plt.tight_layout()
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename_safe = target_key.replace(" ", "_")
    plt.savefig(f"best_model_{filename_safe}.png")
    plt.show()

