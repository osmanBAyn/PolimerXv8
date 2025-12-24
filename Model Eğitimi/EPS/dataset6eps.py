
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

file_name = './EPS/dataset6.csv'

try:
    df = pd.read_csv(file_name)
    df_eps = df[df['property'] == 'eps'].copy()
    
    df_eps['value'] = pd.to_numeric(df_eps['value'], errors='coerce')
    df_eps['smiles'] = df_eps['smiles'].astype(str)
    df_eps = df_eps.dropna(subset=['smiles', 'value'])
    time.sleep(10)
    
    
    def get_features(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            
            desc = [
                Descriptors.TPSA(mol),             # Polarite
                Descriptors.MolLogP(mol),          # Hidrofobiklik
                Descriptors.MolMR(mol),            # Kırılma İndisi 
                Descriptors.MolWt(mol),            # Ağırlık
                Descriptors.NumRotatableBonds(mol) # Esneklik
            ]
            return fp 
        except:
            return None

    df_eps['features'] = df_eps['smiles'].apply(get_features)
    df_eps = df_eps.dropna(subset=['features'])



    X = np.array(df_eps['features'].tolist())
    y = df_eps['value'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1, n_jobs=-1, random_state=42),
        "MLP (Neural Net)": MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=7, weights='distance'), 
        "Ridge Regression": Ridge(alpha=1.0)
    }

    results_data = []
    
    print("-" * 60)
    print(f"{'Model':<20} | {'R2':<8} | {'MAE':<8} | {'RMSE':<8}")
    print("-" * 60)

    best_r2 = -np.inf
    best_name = ""
    best_preds = []

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('model', model)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results_data.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
        print(f"{name:<20} | {r2:<8.4f} | {mae:<8.4f} | {rmse:<8.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_preds = y_pred

    df_results = pd.DataFrame(results_data).sort_values(by="R2", ascending=False)
    
    print("\n" + "="*40)
    print(f"EN İYİ MODEL: {best_name} (R2: {best_r2:.4f})")
    print("="*40)
    print(df_results)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, best_preds, alpha=0.6, color='blue', edgecolors='k', label='Veri Noktaları')
    
    m, M = min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())
    plt.plot([m, M], [m, M], 'r--', lw=3, label='İdeal (y=x)')
    
    plt.title(f"En İyi Model Performansı: {best_name} \n$R^2 = {best_r2:.3f}$", fontsize=24)
    plt.xlabel("Gerçek EPS Değerleri", fontsize=20)
    plt.ylabel("Tahmin Edilen EPS Değerleri", fontsize=20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Hata oluştu: {e}")