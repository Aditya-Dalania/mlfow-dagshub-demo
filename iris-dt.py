import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='Aditya-Dalania', repo_name='mlfow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Aditya-Dalania/mlfow-dagshub-demo.mlflow")  # Set the MLflow tracking server URI

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Decision Tree model
max_depth = 15

mlflow.set_experiment("Iris-dt")

# Start an MLflow run
with mlflow.start_run(run_name='golmal'):
    # Train the Decision Tree model
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = dt.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)


    mlflow.log_param("max_depth", max_depth)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(
    dt,
    name="decision_tree_model",
    serialization_format="cloudpickle",)
    print(f"Accuracy: {accuracy}")
