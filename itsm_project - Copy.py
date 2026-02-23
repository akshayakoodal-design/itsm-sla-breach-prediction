
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import mysql.connector
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1️⃣ Connect
conn = mysql.connector.connect(
    host="your_host",
    user="your_username",
    password="your_password",
    database="your_database"
)

print("✅ Connected successfully!")

# 2️⃣ Load data
query = "SELECT * FROM dataset_list LIMIT 10000;"
df = pd.read_sql(query, conn)

print("✅ Data Loaded Successfully!\n")


# 3️⃣ Convert numeric columns
numeric_cols = [
    'Impact',
    'Urgency',
    'Priority',
    'No_of_Reassignments',
    'Handle_Time_hrs',
    'No_of_Related_Interactions',
    'No_of_Related_Incidents',
    'No_of_Related_Changes'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4️⃣ Convert datetime columns
datetime_cols = [
    'Open_Time',
    'Reopen_Time',
    'Resolved_Time',
    'Close_Time'
]

for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 5️⃣ Create resolution hours
df['Calculated_Resolution_Hours'] = (
    df['Resolved_Time'] - df['Open_Time']
).dt.total_seconds() / 3600

# 6️⃣ Drop missing resolution rows
df = df.dropna(subset=['Calculated_Resolution_Hours'])

# Close connection
conn.close()
print("🔒 Connection Closed")

# 7️⃣ Remove negative and extreme outliers
df = df[df['Calculated_Resolution_Hours'] >= 0]
df = df[df['Calculated_Resolution_Hours'] <= 1000]

# 8️⃣ Create SLA target
df['SLA_Breach'] = (df['Calculated_Resolution_Hours'] > 24).astype(int)

print("\nSLA Distribution:")
print(df['SLA_Breach'].value_counts())

print("\nSLA Percentage:")
print(df['SLA_Breach'].value_counts(normalize=True) * 100)

# 9️⃣ Feature Engineering (time features)
df['Open_Hour'] = df['Open_Time'].dt.hour
df['Open_DayOfWeek'] = df['Open_Time'].dt.dayofweek

# 🔟 Select safe features
features = [
    'Impact',
    'Urgency',
    'Priority',
    'Category',
    'CI_Cat',
    'CI_Subcat',
    'Alert_Status',
    'WBS',
    'Open_Hour',
    'Open_DayOfWeek'
]

X = df[features]
y = df['SLA_Breach']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# ================= MODELING SECTION =================

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=300,

    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 Important Features:")
print(feature_importance.sort_values(ascending=False).head(10))
features = [
    'Impact',
    'Urgency',
    'Priority',
    'No_of_Reassignments',
    'Handle_Time_hrs',
    'No_of_Related_Interactions',
    'No_of_Related_Incidents',
    'No_of_Related_Changes',
    'Category',
    'CI_Cat',
    'CI_Subcat',
    'Alert_Status',
    'WBS',
    'Open_Hour',
    'Open_DayOfWeek'
]
# Plot Top 10 Feature Importance
top10 = feature_importance.sort_values(ascending=False).head(10)

plt.figure()
top10.plot(kind='barh')
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()
plt.show()
