import pandas as pd

df = pd.read_csv("Salary_Data.csv")
'''
print("Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
df = df.dropna(subset=["Salary"])
df = df.dropna()
print(df.isnull().sum())
print("New Shape:", df.shape)
import matplotlib.pyplot as plt
plt.hist(df['Salary'],bins=30)
plt.xlabel(['Salary'])
plt.ylabel(['Frequency'])
plt.title('Salary relation')
plt.show()
plt.scatter(df['Experience'],df['Salary'])
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()'''
#print(df.groupby("Gender")["Salary"].mean())
df = df.dropna()

X = df.drop("Salary", axis=1)
y = df["Salary"]

X = pd.get_dummies(X, drop_first=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
