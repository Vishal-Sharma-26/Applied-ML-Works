import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import os

def plot_confusion_matrix(y_test, y_pred, target_names):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/confusion_matrix.png')
    plt.close()

def plot_feature_importance(X, y, feature_names):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importance = model.feature_importances_
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('static/feature_importance.png')
    plt.close()

def plot_pair_plot(X, y, feature_names, target_names):
    df = pd.DataFrame(X, columns=feature_names)
    df['Species'] = [target_names[i] for i in y]
    sns.pairplot(df, hue='Species', diag_kind='hist')
    plt.savefig('static/pair_plot.png')
    plt.close()