import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_and_eda():
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    print("Fetching dataset from UCI Repository...")
    df = pd.read_csv(url, names=columns, na_values="?")
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df.to_csv("data/heart_clean.csv", index=False)
    
    sns.set_theme(style="whitegrid")
    df.hist(figsize=(12, 10), bins=15)
    plt.tight_layout()
    plt.savefig("artifacts/histograms.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("artifacts/correlation_heatmap.png")
    plt.close()
    print("EDA Visualizations generated in 'artifacts/'.")

if __name__ == '__main__':
    fetch_and_eda()
