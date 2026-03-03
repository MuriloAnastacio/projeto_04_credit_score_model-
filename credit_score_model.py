import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

np.random.seed(42)

# -----------------------------
# 1. Simulação de Base de Crédito
# -----------------------------

n = 10000

data = pd.DataFrame({
    "idade": np.random.randint(18, 70, n),
    "renda_mensal": np.random.gamma(2, 3000, n),
    "tempo_emprego": np.random.randint(0, 20, n),
    "score_bureau": np.random.normal(650, 80, n),
    "percentual_utilizacao": np.random.uniform(0, 1, n)
})

# Variável alvo (inadimplência simulada)
logit = (
    -0.03 * data["idade"]
    -0.0002 * data["renda_mensal"]
    -0.1 * data["tempo_emprego"]
    -0.005 * data["score_bureau"]
    +2 * data["percentual_utilizacao"]
)

prob_default = 1 / (1 + np.exp(-logit))
data["default"] = np.random.binomial(1, prob_default)

# -----------------------------
# 2. Separação Treino/Teste
# -----------------------------

X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. Treinamento Modelo
# -----------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Probabilidades
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 4. Métricas
# -----------------------------

auc = roc_auc_score(y_test, y_pred_prob)
gini = 2 * auc - 1

# KS
ks_stat = ks_2samp(
    y_pred_prob[y_test == 1],
    y_pred_prob[y_test == 0]
).statistic

print(f"AUC: {auc:.4f}")
print(f"Gini: {gini:.4f}")
print(f"KS: {ks_stat:.4f}")

# -----------------------------
# 5. Curva ROC
# -----------------------------

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Modelo de Score")
plt.legend()
plt.show()

# -----------------------------
# 6. Análise de Coeficientes
# -----------------------------

coeficientes = pd.DataFrame({
    "Variavel": X.columns,
    "Coeficiente": model.coef_[0]
}).sort_values(by="Coeficiente", ascending=False)

print("\nCoeficientes do Modelo:")
print(coeficientes)

# -----------------------------
# 7. Simulação de Cutoff
# -----------------------------

cutoff = 0.3
aprovados = y_pred_prob < cutoff

taxa_aprovacao = aprovados.mean()
inadimplencia_aprovados = y_test[aprovados].mean()

print(f"\nTaxa de Aprovação (cutoff={cutoff}): {taxa_aprovacao:.2%}")
print(f"Inadimplência dos Aprovados: {inadimplencia_aprovados:.2%}")
