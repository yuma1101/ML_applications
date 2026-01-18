# ===============================
# 1. ライブラリ読み込み
# ===============================
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===============================
# 2. ダミーデータ作成（SQL想定）
# ===============================
np.random.seed(42)

df = pd.DataFrame({
    "age": np.random.randint(20, 70, 1000),
    "transaction_count": np.random.randint(0, 50, 1000),
    "total_deposit_amount": np.random.randint(0, 5_000_000, 1000),
    "cash_ratio": np.random.rand(1000),
    "nisa_usage_flag": np.random.randint(0, 2, 1000),
    "churn": np.random.randint(0, 2, 1000)  # 1=解約
})
display(df)

# ===============================
# 3. 特徴量 / 目的変数
# ===============================
X = df.drop("churn", axis=1)
y = df["churn"]

# ===============================
# 4. 学習 / テスト分割
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. LightGBM モデル作成 & 学習
# ===============================
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight="balanced",   # 解約案件で重要
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. 予測 & 評価（AUC）
# ===============================
y_pred_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.3f}")
