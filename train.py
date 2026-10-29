import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from data_processing import load_data, split_xy, get_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import numpy as np

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR/"Models"
MODELS_DIR.mkdir(exist_ok=True)


df =load_data()
X,Y = split_xy(df)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 42)

models = {
"LogisticRegression": LogisticRegression(max_iter = 1000,class_weight = "balanced", random_state = 42),
"KNN": KNeighborsClassifier(n_neighbors = 7),
"NaiveBayes": GaussianNB(),
"DecisionTree": DecisionTreeClassifier(max_depth = 5,class_weight = "balanced", random_state = 42),
"RandomForest": RandomForestClassifier(n_estimators=200, class_weight = "balanced", random_state = 42),
"SVM": SVC( probability = True, class_weight = "balanced"),
}

results = []
trained = {}
thresholds_dict = {}
for name,model in models.items():
    pipe = Pipeline([
        ("Processor",get_preprocessor()),
        ("Models",model)
 ])
    pipe.fit(X_train,Y_train)
    probabilities = pipe.predict_proba(X_test)[:, 1]

  # 2. حساب أفضل Threshold يحقق أعلى F1-Score
    precisions, recalls, thresholds = precision_recall_curve(Y_test, probabilities)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

  # 3. عمل التوقعات النهائية بناءً على الـ Threshold الجديد
    custom_predictions = (probabilities >= best_threshold).astype(int)

    predictions = pipe.predict(X_test)
    results.append(
{
        "Model":name,
        "Accuracy":accuracy_score(Y_test,custom_predictions),
        "F1":f1_score(Y_test,custom_predictions),
        "Precision":precision_score(Y_test,custom_predictions),
        "Recall":recall_score(Y_test,custom_predictions),
        "Best_Threshold": best_threshold,
    })
    trained[name] = pipe
    print(f"{name} is done")
    trained[name] = pipe
    thresholds_dict[name] = best_threshold
    print(f"{name} is done (Best Threshold: {best_threshold:.2f})")

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n", results_df.round(3).to_string(index=False))

best_name = results_df.loc[0, "Model"]
print(f"\nBest model (by F1_score): {best_name}"
  f"\nBest threshold: {thresholds_dict[best_name]:.2f}")

results_df.to_csv(MODELS_DIR / "results.csv", index=False)
joblib.dump((trained[best_name],best_threshold), MODELS_DIR / "churn_model.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(best_name)