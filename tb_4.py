import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, make_scorer, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. Load and Clean Dataset ---
# Ensure your file 'final_dataset.txt' is in the same directory
df = pd.read_csv('final_dataset.txt', sep='\t')
df.columns = df.columns.str.strip()

X = df.iloc[:, 1:-1] 
y = df.iloc[:, -1]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 2. Feature Selection Strategies ---
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)

class ACOFeatureSelection:
    def fit(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        # Ensure we pick at least some features
        self.selected_features = np.where(importances > np.mean(importances))[0]
        return X[:, self.selected_features]

aco = ACOFeatureSelection()
X_train_aco = aco.fit(X_train_scaled, y_train)

strategies = {"PCA": X_train_pca, "ACO": X_train_aco}

# --- 3. Models and Hyperparameters ---
models = {
    'RF': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}),
    'SVM': (SVC(probability=True, random_state=42), {'C': [0.1, 1, 10], 'kernel': ['rbf']}),
    'LR': (LogisticRegression(max_iter=5000, random_state=42), {'C': [0.1, 1, 10]}),
    'XGBoost': (XGBClassifier(eval_metric='logloss', random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.1]})
}

results_list = []

# --- 4. Training Loop ---
for strat_name, X_data in strategies.items():
    for model_name, (base_clf, params) in models.items():
        print(f"Running {model_name} with {strat_name}...")
        grid = GridSearchCV(base_clf, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_data, y_train)
        
        results_list.append({
            'Strategy': strat_name, 
            'Classifier': model_name,
            'Hyperparameters': grid.best_params_,
            'Mean_Accuracy': grid.best_score_
        })

results_df = pd.DataFrame(results_list)
best_config = results_df.loc[results_df['Mean_Accuracy'].idxmax()]

# --- 5. Re-train the Champion ---
print(f"\nChampion: {best_config['Classifier']} via {best_config['Strategy']}")

X_train_final = strategies[best_config['Strategy']]
X_test_scaled = scaler.transform(X_test)

# Match test data to the winning strategy
if best_config['Strategy'] == "PCA":
    X_test_final = pca.transform(X_test_scaled)
else:
    X_test_final = X_test_scaled[:, aco.selected_features]

final_model = model_map = {
    'RF': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'LR': LogisticRegression(max_iter=5000, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}[best_config['Classifier']]

final_model.set_params(**best_config['Hyperparameters'])
final_model.fit(X_train_final, y_train)
y_pred = final_model.predict(X_test_final)

# --- 6. Plotting: CM and Scatter ---
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
ax1.set_title("Confusion Matrix")

# Scatter Plot (Top 2 Dimensions)
if best_config['Strategy'] == "PCA":
    plot_data = X_test_final[:, :2]
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
else:
    # Use first two selected features for ACO scatter
    plot_data = X_test_final[:, :2]
    feat_names = X.columns[aco.selected_features]
    ax2.set_xlabel(feat_names[0])
    ax2.set_ylabel(feat_names[1])

for i, class_label in enumerate(le.classes_):
    mask = y_test == i
    ax2.scatter(plot_data[mask, 0], plot_data[mask, 1], label=class_label, alpha=0.6)
ax2.set_title(f"Class Distribution ({best_config['Strategy']})")
ax2.legend()
plt.tight_layout()
plt.show()

# --- 7. Save Top 50 Biomarkers ---
print("\nSaving Top 50 Biomarkers...")
if best_config['Strategy'] == "PCA":
    # Link PCA components back to original features
    loadings = pd.DataFrame(np.abs(pca.components_[:3].T), index=X.columns, columns=['PC1', 'PC2', 'PC3'])
    top_50 = loadings.sum(axis=1).nlargest(50)
else:
    feat_names = X.columns[aco.selected_features]
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
    elif hasattr(final_model, 'coef_'):
        importances = np.mean(np.abs(final_model.coef_), axis=0)
    else:
        # Fallback to the initial ACO Random Forest importances
        temp_rf = RandomForestClassifier().fit(X_train_final, y_train)
        importances = temp_rf.feature_importances_
    
    top_50 = pd.Series(importances, index=feat_names).nlargest(50)

top_50.to_csv('top_50_biomarkers.csv')
print("File 'top_50_biomarkers.csv' created.")
print(classification_report(y_test, y_pred, target_names=le.classes_))