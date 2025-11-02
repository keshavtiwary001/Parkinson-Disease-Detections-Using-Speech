# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from joblib import dump, load
# from filters.praat import Praat

# def regressor(df):
#     # Regressor to predict motor_UPDRS
#     label = df['motor_UPDRS']
#     features = df.drop(columns=["motor_UPDRS", "total_UPDRS", "subject#", "test_time"])

#     x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

#     model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
#     model.fit(x_train, y_train)
#     print(model.score(x_test, y_test))

# def classifier(df):
#     label = df['status']
#     features = df.drop(columns=["status", "DFA", "PPE", "RPDE"]) # IMPORTANT - praat does not have DFA, PPE, RPDE yet so we drop them for now
#     x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
#     model.fit(x_train, y_train)
#     print(model.score(x_test, y_test))
#     dump(model, "models/randomforest.joblib")

# def classify_using_saved_model(audio_sample, is_cloud=True):
#     model = load("models/randomforest.joblib")
#     praat = Praat()
#     features = praat.getFeatures(audio_sample, 75, 200)
#     if is_cloud:
#         import os
#         BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         TMP_DIR = os.path.join(BASE_DIR, "tmp")
#         os.makedirs(TMP_DIR, exist_ok=True)
#         praat.generateSpectrogram(audio_sample, TMP_DIR)

#     else:
#         praat.generateSpectrogram(audio_sample)
#         df = pd.DataFrame([features])
#         return model.predict(df)

# def test_multiple_classifiers(df):
#     log_reg_params = [{"C":0.01}, {"C":0.1}, {"C":1}, {"C":10}]
#     dec_tree_params = [{"criterion": "gini"}, {"criterion": "entropy"}]
#     rand_for_params = [{"criterion": "gini"}, {"criterion": "entropy"}]
#     kneighbors_params = [{"n_neighbors":3}, {"n_neighbors":5}]
#     naive_bayes_params = [{}]
#     svc_params = [{"C":0.01}, {"C":0.1}, {"C":1}, {"C":10}]
#     modelclasses = [
#         ["log regression", LogisticRegression, log_reg_params],
#         ["decision tree", DecisionTreeClassifier, dec_tree_params],
#         ["random forest", RandomForestClassifier, rand_for_params],
#         ["k neighbors", KNeighborsClassifier, kneighbors_params],
#         ["naive bayes", GaussianNB, naive_bayes_params],
#         ["support vector machines", SVC, svc_params]
#     ]
#     insights = []
#     label = df['status']
#     features = df.drop(columns=["status"])

#     x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
#     for modelname, Model, params_list in modelclasses:
#         for params in params_list:
#             model = Model(**params)
#             model.fit(x_train, y_train)
#             score = model.score(x_test, y_test)      
#             insights.append((modelname, model, params, score))

#     insights.sort(key=lambda x:x[-1], reverse=True)
#     for modelname, model, params, score in insights:
#         print(modelname, params, score)

# df1 = pd.read_csv("data/parkinsons_timeseries.csv")
# df2 = pd.read_csv("data/healthy_and_unhealthy.csv")
# df3 = pd.read_csv("data/healthy.csv")
# df4 = pd.read_csv("data/unhealthy.csv")

# df3.rename(columns={"Unnamed: 0":"status"}, inplace=True)
# df4.rename(columns={"Unnamed: 0":"status"}, inplace=True)
# df3["status"] = 0
# df4["status"] = 1

# df1["motor_UPDRS"] = df1["motor_UPDRS"].apply(lambda x: 1 if x > 12 else 0)
# df1.rename(columns={"motor_UPDRS": "status"}, inplace=True)
# df2.rename(columns={"MDVP:Jitter(%)":"Jitter(%)", "MDVP:Jitter(Abs)":"Jitter(Abs)", "MDVP:RAP":"Jitter:RAP", "MDVP:PPQ":"Jitter:PPQ5",
#                     "MDVP:Shimmer":"Shimmer", "MDVP:Shimmer(dB)":"Shimmer(dB)", "MDVP:APQ":"Shimmer:APQ11", }, inplace=True)

# df3.rename(columns={"MDVP:Jitter(%)":"Jitter(%)", "MDVP:Jitter(Abs)":"Jitter(Abs)", "MDVP:RAP":"Jitter:RAP", "MDVP:PPQ":"Jitter:PPQ5",
#                     "MDVP:Shimmer":"Shimmer", "MDVP:Shimmer(dB)":"Shimmer(dB)", "MDVP:APQ":"Shimmer:APQ11", }, inplace=True)

# df4.rename(columns={"MDVP:Jitter(%)":"Jitter(%)", "MDVP:Jitter(Abs)":"Jitter(Abs)", "MDVP:RAP":"Jitter:RAP", "MDVP:PPQ":"Jitter:PPQ5",
#                     "MDVP:Shimmer":"Shimmer", "MDVP:Shimmer(dB)":"Shimmer(dB)", "MDVP:APQ":"Shimmer:APQ11", }, inplace=True)

# df2.drop(columns=["name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "spread1", "spread2", "D2", "NHR"], inplace=True)
# df1.drop(columns=["total_UPDRS", "subject#", "test_time", "age", "sex", "NHR"], inplace=True)

# df = pd.concat([df1, df2, df3, df4])

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from joblib import dump, load
from filters.praat import Praat


# --- Ensure models directory exists ---
os.makedirs("models", exist_ok=True)


# -------------------------------
# 1️⃣ RandomForest Regressor (optional)
# -------------------------------
def regressor(df):
    """Train a RandomForest regressor for motor_UPDRS."""
    label = df['motor_UPDRS']
    features = df.drop(columns=["motor_UPDRS", "total_UPDRS", "subject#", "test_time"], errors="ignore")

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(x_train, y_train)
    print("Regressor R² Score:", model.score(x_test, y_test))


# -------------------------------
# 2️⃣ RandomForest Classifier (main model)
# -------------------------------
def classifier(df):
    """Train and save a RandomForest classifier for Parkinson detection."""
    label = df['status']
    features = df.drop(columns=["status", "DFA", "PPE", "RPDE"], errors="ignore")

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(x_train, y_train)
    print("Classifier Accuracy:", model.score(x_test, y_test))

    # Save model
    dump(model, "models/randomforest.joblib")
    print("✅ Model saved to models/randomforest.joblib")


# -------------------------------
# 3️⃣ Predict using saved RandomForest model
# -------------------------------
def classify_using_saved_model(audio_sample, is_cloud=True):
    """Classify an audio sample using saved RandomForest model + Praat features."""
    model = load("models/randomforest.joblib")
    praat = Praat()
    features = praat.getFeatures(audio_sample, 75, 200)

    if is_cloud:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TMP_DIR = os.path.join(BASE_DIR, "tmp")
        os.makedirs(TMP_DIR, exist_ok=True)
        praat.generateSpectrogram(audio_sample, TMP_DIR)
    else:
        praat.generateSpectrogram(audio_sample)

    df = pd.DataFrame([features])
    prediction = model.predict(df)
    print("Predicted label:", prediction[0])
    return prediction


# -------------------------------
# 4️⃣ Compare multiple classifiers
# -------------------------------
def test_multiple_classifiers(df):
    """Test multiple models with various hyperparameters."""
    log_reg_params = [{"C": 0.01}, {"C": 0.1}, {"C": 1}, {"C": 10}]
    dec_tree_params = [{"criterion": "gini"}, {"criterion": "entropy"}]
    rand_for_params = [{"criterion": "gini"}, {"criterion": "entropy"}]
    kneighbors_params = [{"n_neighbors": 3}, {"n_neighbors": 5}]
    naive_bayes_params = [{}]
    svc_params = [{"C": 0.01}, {"C": 0.1}, {"C": 1}, {"C": 10}]

    modelclasses = [
        ["Logistic Regression", LogisticRegression, log_reg_params],
        ["Decision Tree", DecisionTreeClassifier, dec_tree_params],
        ["Random Forest", RandomForestClassifier, rand_for_params],
        ["K-Nearest Neighbors", KNeighborsClassifier, kneighbors_params],
        ["Naive Bayes", GaussianNB, naive_bayes_params],
        ["Support Vector Machine", SVC, svc_params]
    ]

    insights = []
    label = df['status']
    features = df.drop(columns=["status"], errors="ignore")

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

    for modelname, Model, params_list in modelclasses:
        for params in params_list:
            model = Model(**params)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            insights.append((modelname, params, score))

    insights.sort(key=lambda x: x[-1], reverse=True)
    print("\n--- Classifier Performance Comparison ---")
    for modelname, params, score in insights:
        print(f"{modelname:25s} {params} -> Accuracy: {score:.4f}")


# -------------------------------
# 5️⃣ Load and preprocess datasets
# -------------------------------
df1 = pd.read_csv("data/parkinsons_timeseries.csv")
df2 = pd.read_csv("data/healthy_and_unhealthy.csv")
df3 = pd.read_csv("data/healthy.csv")
df4 = pd.read_csv("data/unhealthy.csv")

# Fix unnamed column & add status labels
df3.rename(columns={"Unnamed: 0": "status"}, inplace=True)
df4.rename(columns={"Unnamed: 0": "status"}, inplace=True)
df3["status"] = 0
df4["status"] = 1

# Convert UPDRS to binary status
df1["motor_UPDRS"] = df1["motor_UPDRS"].apply(lambda x: 1 if x > 12 else 0)
df1.rename(columns={"motor_UPDRS": "status"}, inplace=True)

# Standardize column names
rename_map = {
    "MDVP:Jitter(%)": "Jitter(%)",
    "MDVP:Jitter(Abs)": "Jitter(Abs)",
    "MDVP:RAP": "Jitter:RAP",
    "MDVP:PPQ": "Jitter:PPQ5",
    "MDVP:Shimmer": "Shimmer",
    "MDVP:Shimmer(dB)": "Shimmer(dB)",
    "MDVP:APQ": "Shimmer:APQ11"
}
for df in [df2, df3, df4]:
    df.rename(columns=rename_map, inplace=True)

# Drop unnecessary columns
df2.drop(columns=["name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
                  "spread1", "spread2", "D2", "NHR"], inplace=True, errors="ignore")
df1.drop(columns=["total_UPDRS", "subject#", "test_time", "age", "sex", "NHR"],
         inplace=True, errors="ignore")

# Merge all datasets
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

print(f"✅ Final dataset shape: {df.shape}")
print(df.head())

# -------------------------------
# 6️⃣ Train and save model
# -------------------------------
classifier(df)
# Optional: test_multiple_classifiers(df)
