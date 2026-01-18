

# 1. データの準備（ダミーデータ）
# 年齢、資産残高、取引回数という3つの特徴量を持つ5人分のデータ
data = {
    'age': [25, 45, 32, 60, 22],
    'balance': [100, 5000, 800, 10000, 50],
    'transactions': [2, 20, 5, 30, 1]
}
X = pd.DataFrame(data)
display(X)

# 目的変数（1: 解約しそう、0: 継続しそう）
y = [0, 1, 0, 1, 0]

# 2. モデルの作成と学習
# 特徴量(X)と正解ラベル(y)をモデルに覚えさせる
model = lgb.LGBMClassifier()
model.fit(X, y)

# 3. 新しい顧客の予測
# 例：30歳、残高2000、取引回数10回の顧客はどうなるか？
new_customer = pd.DataFrame([[30, 2000, 10]], columns=['age', 'balance', 'transactions'])
prediction = model.predict(new_customer)
prob = model.predict_proba(new_customer)

print(f"予測結果: {'解約リスクあり' if prediction[0] == 1 else '継続'}")
print(f"解約の確率: {prob[0][1] * 100}%")