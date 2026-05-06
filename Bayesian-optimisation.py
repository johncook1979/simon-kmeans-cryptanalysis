pip install pandas numpy scikit-learn optuna

import random
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import optuna
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna.visualization as vis


def kmeans_bool_round(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.round()
    labels = []
    for x in X:
        distances = [np.linalg.norm(x - c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels), centroids


df = pd.read_csv('processed-p2-not--rounded.csv')
X = df[['alpha_pow_2', 'beta_pow_2', 'gam_pow_2']].astype(int)

if 'inputs_match' in df.columns:
    y = df['inputs_match'].astype(int)
else:
    y = None



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- Objective function ---
def objective(trial):
    model_name = trial.suggest_categorical("model", ["DecisionTree", "RandomForest", "SVM", "KNN", "DNN", "CNN", "KMeans"])

    if model_name == "DecisionTree":
        max_depth = trial.suggest_int("max_depth", 1, 10)
        clf = DecisionTreeClassifier(max_depth=max_depth)
        score = cross_val_score(clf, X, y, cv=3).mean()
        return score

    elif model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 16)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        score = cross_val_score(clf, X, y, cv=3).mean()
        return score

    elif model_name == "SVM":
        C = trial.suggest_loguniform("C", 1e-2, 1e2)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        clf = SVC(C=C, kernel=kernel)
        score = cross_val_score(clf, X, y, cv=3).mean()
        return score

    elif model_name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 10)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        score = cross_val_score(clf, X, y, cv=3).mean()
        return score

    elif model_name == "DNN":
        # Keep it tiny for Colab
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        model.add(layers.Dense(trial.suggest_int("units", 4, 16), activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=3, verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        return acc

    elif model_name == "CNN":
        # Fake CNN for tabular: reshape to (samples, features, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        X_train_cnn = X_train.reshape((-1, X_train.shape[1], 1))
        X_test_cnn = X_test.reshape((-1, X_test.shape[1], 1))

        model = keras.Sequential()
        model.add(layers.Conv1D(filters=trial.suggest_int("filters", 4, 16), kernel_size=2, activation='relu', input_shape=(X_train.shape[1],1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_cnn, y_train, epochs=3, verbose=0)
        _, acc = model.evaluate(X_test_cnn, y_test, verbose=0)
        return acc

    elif model_name == "KMeans":
        n_clusters = trial.suggest_int("n_clusters", 2, 8)
        labels, centroids = kmeans_bool_round(X_scaled, n_clusters)
        score = silhouette_score(X_scaled, labels)
        return score
 --- Run study ---
# Save to local file instead of in-memory
study = optuna.create_study(
    study_name="my_supervised_unsupervised_search",
    direction="maximize",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True
)

study.optimize(objective, n_trials=50) # keep it small in Colab!

print("Best trial:")
print(study.best_trial.params)


fig1 = vis.plot_optimization_history(study, target_name="Trial Value")

# Modify the y-axis range
fig1.update_layout(yaxis_range=[0.95, 1.005])
fig1.update_layout(xaxis_range=[0, 45])
fig1.update_layout(title=None) # Remove the title

# Get the highest result for each model from the study
trials_df = study.trials_dataframe()

# Filter for completed trials and ensure necessary parameter column exists
# Use 'params_model' directly from the dataframe
completed_trials_df = trials_df[trials_df['state'] == 'COMPLETE'].copy()

# Check if 'params_model' column exists and filter out rows where it's NaN
if 'params_model' in completed_trials_df.columns:
    completed_trials_df = completed_trials_df.dropna(subset=['params_model']).copy() # Use .copy() to avoid SettingWithCopyWarning
    highest_results_per_model = completed_trials_df.groupby('params_model')['value'].max().to_dict()
else:
    # If 'params_model' column doesn't exist, we cannot group by model
    highest_results_per_model = {}


annotated_models = set()
vertical_offset_base = [30, 70, 110]  # Base vertical offset for annotations
horizontal_offset_base = 0 # Base horizontal offset for annotations

# Add annotations only to the highest trial.value of each model and only once for each model
for trial in study.trials:

    # Add checks for trial state, value, and parameters
    if (trial.state.is_finished() and
        trial.value is not None and
        trial.params is not None and
        isinstance(trial.params, dict)):

        # Safely get the model name using .get()
        model_name = trial.params.get('model')

        # Proceed only if model_name is available and not None
        if model_name is not None:
            # Check if this trial's value is the highest for its model and the model hasn't been annotated yet
            # Use a small tolerance for floating point comparison
            if model_name in highest_results_per_model and abs(trial.value - highest_results_per_model[model_name]) < 1e-9 and model_name not in annotated_models:

                value_text = f"{trial.value:.12f}"
                annotation_text = f"Trial {trial.number}<br>Value: {value_text}<br>Model: {model_name}"

                bk_col = "rgba(255, 255, 255, 0.8)"
                # Determine the vertical and horizontal offset based on the number of models already annotated
                if model_name == "KNN":
                    vertical_offset = -60
                    horizontal_offset = 60
                elif model_name == "RandomForest":
                    vertical_offset = 60
                    horizontal_offset = -20
                elif model_name == "DNN":
                    vertical_offset = -60
                    horizontal_offset = -70
                elif model_name == "SVM":
                    vertical_offset = 140
                    horizontal_offset = -80
                elif model_name == "CNN":
                    vertical_offset = 60
                    horizontal_offset = 60
                elif model_name == "KMeans":
                    vertical_offset = 120
                    horizontal_offset = 40
                    bk_col = "rgba(39, 245, 200, 0.5)"
                else:
                    vertical_offset = -60
                    horizontal_offset = 100

                fig1.add_annotation(
                    x=trial.number,
                    y=trial.value,
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=3,
                    ax=horizontal_offset,
                    ay=vertical_offset,
                    font=dict(
                        size=16,
                        color="black"
                    ),
                    bgcolor=bk_col
                )
                annotated_models.add(model_name) # Mark the model as annotated

fig1.update_layout(
    xaxis=dict(
        title_text="Trial Number",
        title_font=dict(size=18, color="black"),
        title_standoff=20,
        tickfont=dict(size=16)
    ),
    yaxis=dict(
        title_text="ML Model Validation Value",
        title_font=dict(size=18, color="black"),
        title_standoff=20,
        tickfont=dict(size=16)
    ),
    # Added this to enlarge the legend font
    legend=dict(font=dict(size=18))
)

# Display the modified plot
fig1.show()


# Display other plots as before
vis.plot_param_importances(study).show()
vis.plot_parallel_coordinate(study).show()
vis.plot_slice(study).show()


# Get the trials DataFrame from the study
trials_df = study.trials_dataframe()

# Create a new DataFrame with the desired columns
results_table = pd.DataFrame()

# Add Trial Number
results_table['Trial Number'] = trials_df['number']

# Add ML Model Name
# Access 'params_model' directly from the dataframe
results_table['ML Model Name'] = trials_df['params_model']

# Add Evaluation Value (already in 'value' column)
results_table['Evaluation Value'] = trials_df['value']

# Determine Evaluation Metric based on ML Model Name
# You can define a mapping from model name to metric
metric_mapping = {
    'DecisionTree': 'Accuracy (Cross-validation)',
    'RandomForest': 'Accuracy (Cross-validation)',
    'SVM': 'Accuracy (Cross-validation)',
    'KNN': 'Accuracy (Cross-validation)',
    'DNN': 'Accuracy (Test Set)',
    'CNN': 'Accuracy (Test Set)',
    'KMeans': 'Silhouette Score'
}

# Map the model names to their corresponding metrics
results_table['Evaluation Metric'] = results_table['ML Model Name'].map(metric_mapping)

# Format Evaluation Value to 12 decimal places for display
# Use pandas display options within a context manager to limit the scope of the change
with pd.option_context('display.float_format', '{:.12f}'.format):
    # Display the resulting table
    display(results_table)
