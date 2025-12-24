import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
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
        mol = Chem.MolFromSmiles(str(smiles_str))
        if mol is None: return None
        
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
        
        desc = [
            Descriptors.MolMR(mol),            # Molar Refractivity 
            Descriptors.MolWt(mol),            # Ağırlık
            Descriptors.MolLogP(mol),          # Hidrofobiklik
            Descriptors.TPSA(mol),             # Polarite
            Descriptors.NumRotatableBonds(mol), # Esneklik
            Descriptors.HeavyAtomCount(mol)    # Atom sayısı
        ]
        return fp + desc
    except:
        return None

def load_data(filename, task_name):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"   HATA: {filename} bulunamadı.")
        return None, None

    if 'task' in df.columns:
        df_task = df[df['task'] == task_name].copy()
    else:
        df_task = df.copy()

    rename_map = {'prompt': 'smiles', 'target': 'value', 'SMILES': 'smiles', 'Target': 'value'}
    df_task = df_task.rename(columns=rename_map)

    df_task['value'] = pd.to_numeric(df_task['value'], errors='coerce')
    df_task['smiles'] = df_task['smiles'].astype(str)
    df_task = df_task.dropna(subset=['smiles', 'value'])
    df_task = df_task.drop_duplicates(subset=['smiles'], keep='first')
    
    if len(df_task) == 0:
        print(f"   UYARI: Veri bulunamadı.")
        return None, None

    df_task['features'] = df_task['smiles'].apply(get_features)
    df_task = df_task.dropna(subset=['features'])

    time.sleep(10)
    X = np.array(df_task['features'].tolist())
    y = df_task['value'].values
    return X, y


task_name = 'refractive index'

models = {
    "RandomForest": RandomForestRegressor(n_estimators=530, max_depth=20, min_samples_split=9, min_samples_leaf=1, max_features=1, random_state=42, n_jobs=-1),
    # "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42),
    # "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1, n_jobs=-1, random_state=42),
    # "KNN": KNeighborsRegressor(n_neighbors=5),
    # "Ridge": Ridge(alpha=1.0),
    # "MLP": MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42)
}

print(f"\n--- {task_name.upper()} TAHMİNİ: 6 MODEL KARŞILAŞTIRMASI ---")
print("(Eğitim: dataset3.csv | Test: dataset4.csv)\n")


X_train, y_train = load_data('./Refractive Index/dataset3.csv', task_name)
X_test, y_test = load_data('./Refractive Index/dataset4.csv', task_name)

if X_train is not None and X_test is not None:
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
        if name in ["RandomForest", "XGBoost", "LightGBM"]:
            X_tr, X_te = X_train, X_test
        else:
            X_tr, X_te = X_train_scaled, X_test_scaled
        
        model.fit(X_tr, y_train)
        
        y_pred = model.predict(X_te)
        
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

    print(f"\n   >>>: {best_model_name} (R2: {best_r2:.4f})")
    
    df_res = pd.DataFrame(results_data).sort_values(by="R2", ascending=False)
    print("\n", df_res)

    plt.figure(figsize=(8, 8))

    plt.scatter(y_test, best_y_pred, alpha=0.6, color='blue', edgecolors='k', label='Veri Noktaları')

    min_val = min(min(y_test), min(best_y_pred))
    max_val = max(max(y_test), max(best_y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Mükemmel Tahmin (y=x)')

    plt.title(f'En İyi Model Performansı: {best_model_name}\n($R^2$: {best_r2:.3f})', fontsize=24)
    plt.xlabel('Gerçek Refractive Index Değerleri', fontsize=20)
    plt.ylabel('Tahmin Edilen Refractive Index Değerleri', fontsize=20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

else:
    print("\nVeri yüklenemedi.")