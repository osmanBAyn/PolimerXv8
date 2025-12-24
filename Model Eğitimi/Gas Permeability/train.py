import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

url = "https://raw.githubusercontent.com/jsunn-y/PolymerGasMembraneML/main/datasets/datasetA_imputed_all.csv"

try:
    df = pd.read_csv(url)
except:
    url = "https://raw.githubusercontent.com/jsunn-y/PolymerGasMembraneML/master/datasets/datasetA_imputed_all.csv"
    df = pd.read_csv(url)

target_gas = 'CO2'
if target_gas not in df.columns:
    target_gas = [c for c in df.columns if 'CO2' in c.upper()][0]

df_clean = df.dropna(subset=['Smiles', target_gas])
df_clean = df_clean[df_clean[target_gas] > 0]

time.sleep(10)

X_smiles = df_clean['Smiles'].values
y = np.log10(df_clean[target_gas].values) 

def get_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HallKierAlpha(mol)
        ])
        return np.concatenate((fp, desc))
    except:
        return None

X_list = [get_features(s) for s in X_smiles]

X = []
y_final = []
for i, feat in enumerate(X_list):
    if feat is not None:
        X.append(feat)
        y_final.append(y[i])

X = np.array(X)
y = np.array(y_final)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=10),
    "MLP (Neural Net)": MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1, n_jobs=-1, random_state=42)
}

results_data = []
best_r2 = -np.inf
best_model_name = ""
best_y_pred = []


for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('regressor', model)
    ])
    
    try:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results_data.append({"Model": name, "R2 Score": r2, "RMSE": rmse})
        print(f"--> {name:<20} | R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_y_pred = y_pred
            
    except Exception as e:
        print(f"--> {name} HATA: {e}")

results_df = pd.DataFrame(results_data).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)

print("\n" + "="*45)
print("KARŞILAŞTIRMA TABLOSU")
print("="*45)
print(results_df)

plt.figure(figsize=(8, 8))

plt.scatter(y_test, best_y_pred, alpha=0.6, color='blue', edgecolors='k', label='Veri Noktaları')

min_val = min(min(y_test), min(best_y_pred))
max_val = max(max(y_test), max(best_y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Mükemmel Tahmin (y=x)')

plt.title(f'En İyi Model Performansı: {best_model_name}\n($R^2$: {best_r2:.3f})', fontsize=24)
plt.xlabel('Gerçek CO2 Geçirgenliği Değerleri', fontsize=20)
plt.ylabel('Tahmin Edilen CO2 Geçirgenliği Değerleri', fontsize=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="R2 Score", y="Model", data=results_df, palette="magma")
plt.title("Modellerin R2 Performans Sıralaması")
plt.xlim(0, 1.0)
plt.show()