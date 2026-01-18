import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test  =pd.read_csv('test.csv')
display(train.head())
display(test.head())


display(train.info())
display(train.describe())
display(train.describe(include="object"))
display(train.isnull().sum())
display(train.shape)


display(test.info())
display(test.describe())
display(test.describe(include="object"))
display(test.isnull().sum())
display(test.shape)

#chun 解約(チャーン)
target = "churn"
sns.countplot(x=target,data=train)
train.groupby(target).count()["customer_id"]
#「1」が解約した人
# 20.5%が解約


#credit_score
fig = plt.subplots(figsize=(20,10))
sns.histplot(x="credit_score",hue="churn",data=train,kde=True)
train.groupby("credit_score").count()["customer_id"]

#country ３カ国
fig = plt.subplots(figsize=(20,10))
sns.countplot(x="country",data=train,hue="churn")
train[["country","churn"]].groupby("country").mean()
# Germanyの解約率が他の2倍くらい近く高い
# Franceをベースにone-hot encoding

# gender
fig = plt.subplots(figsize=(20,10))
sns.countplot(x="gender",data=train,hue="churn")
train[["gender","churn"]].groupby("gender").mean()
# 女性の方が解約率は高い
# maleを1、Femaleを0にダミー変換


# age
fig = plt.subplots(figsize=(20,10))
sns.histplot(x="age",hue="churn",data=train,kde=True)
train.groupby("churn",as_index=False).describe()[["churn","age"]]
# 基本的に年齢が上がるほど解約率が高くなり、40後半〜60までは解約率が高い
# 年齢に加えて、"45~60"かどうか判定する特徴量を加えても良さそう


# tenure
target ="tenure"
sns.histplot(x=target, hue="churn",data=train)
train.groupby("churn",as_index=False).describe()[["churn",target]]
#ほとんど差なし


# balance(残高)

target = "balance"
sns.histplot(x=target,hue="churn",data=train)
train.groupby("churn",as_index=False).describe()[["churn",target]]
#残高が0とそうでない人に二分している


# balance＝0とそれ以外の解約率
print("balanceが0の解約率:",train.query('balance == 0')["churn"].mean())
print("balanceが0以上の解約率:",train.query('balance != 0')["churn"].mean())

# balanceが0だと解約率が半分になる
# balanceが0かそうでないかのdummy変換をした方が良い


# producst_number
target = "products_number"
sns.countplot(x="products_number",hue="churn",data=train)
print(train.groupby(target).count()["customer_id"])
print(train[["churn",target]].groupby(target).mean())
# 2が解約率低めだが、3,4になると逆に増えている
# カテゴリー変数として扱い、1をベースに２、3以上をそれぞれone-hot encoding


# credit_card
target = "credit_card"
sns.countplot(x=target,hue="churn",data=train)
train[["churn",target]].groupby(target).mean()


# active_member
target = "active_member"
sns.countplot(x=target,hue="churn",data=train)
train[["churn",target]].groupby(target).mean()

# estimated_salary
target = "estimated_salary"
sns.histplot(x=target,hue="churn",data=train)
train.groupby("churn",as_index=False).describe()[["churn",target]]




#　データの前処理

#edaの結果を元に、特徴量を適切な形に変換
df_train = train.copy() 
df_test = test.copy() 

#customer_idの削除
df_train = df_train.drop("customer_id",axis=1)
df_test = df_test.drop("customer_id",axis=1)

# dummy_train = df_train
# df_dummy = pd.get_dummies(dummy_train)
# display(df_dummy)


#country Franceをベースにone-hot encoding
df_train["country_Germany"] = df_train["country"].apply(lambda x : 1 if x == "Germany" else 0)
df_train["country_Spain"] = df_train["country"].apply(lambda x : 1 if x == "Spain" else 0)
df_train = df_train.drop("country",axis=1)
df_test["country_Germany"] = df_test["country"].apply(lambda x : 1 if x == "Germany" else 0)
df_test["country_Spain"] = df_test["country"].apply(lambda x : 1 if x == "Spain" else 0)
df_test = df_test.drop("country",axis=1)

# gender male=1,Female=0
df_train["gender"] = df_train["gender"].apply(lambda x : 1 if x == "Male" else 0)
df_test["gender"] = df_test["gender"].apply(lambda x : 1 if x == "Male" else 0)

# age "45~60"かどう判定する特徴量を追加
df_train["age45-60"] = df_train["age"].apply(lambda x : 1 if x>=45 and x<=60 else 0)
df_test["age45-60"] = df_test["age"].apply(lambda x : 1 if x>=45 and x<=60 else 0)

#balance 0かそうでないかでdummy変換
df_train["balance"] = df_train["balance"].apply(lambda x : 1 if x > 0 else 0)
df_test["balance"] = df_test["balance"].apply(lambda x : 1 if x > 0 else 0)

# product_number 1をbaseに2,3以上をそれぞれone-hot encoding
df_train["products_number_2"] = df_train["products_number"].apply(lambda x:1 if x == 2 else 0)
df_train["products_number_over3"] = df_train["products_number"].apply(lambda x:1 if x > 2 else 0)
df_train = df_train.drop("products_number",axis=1)
df_test["products_number_2"] = df_test["products_number"].apply(lambda x:1 if x == 2 else 0)
df_test["products_number_over3"] = df_test["products_number"].apply(lambda x:1 if x > 2 else 0)
df_test = df_test.drop("products_number",axis=1)

display(df_train.info())
display(df_train.describe())
display(df_test.info())
display(df_test.describe())








