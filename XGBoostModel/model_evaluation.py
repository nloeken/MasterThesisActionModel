from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_action, model_success, X_model1_test, X_model2_test, y_model1_test, y_model2_test, le_action):
    # --- Modell 1: Next Action Category ---
    y_pred_action = model_action.predict(X_model1_test)

    print("=== Model 1: Next Action Category ===")
    print("Accuracy:", accuracy_score(y_model1_test, y_pred_action))
    print("F1 Score (weighted):", f1_score(y_model1_test, y_pred_action, average='weighted'))
    print(classification_report(y_model1_test, y_pred_action, target_names=le_action.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_model1_test, y_pred_action)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_action.classes_, yticklabels=le_action.classes_)
    plt.title("Confusion Matrix – Next Action Category")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # --- Modell 2: Success Probability ---
    y_pred_success = model_success.predict(X_model2_test)

    print("\n=== Model 2: Success Probability ===")
    print("Accuracy:", accuracy_score(y_model2_test, y_pred_success))
    print("F1 Score (weighted):", f1_score(y_model2_test, y_pred_success, average='weighted'))
    print(classification_report(y_model2_test, y_pred_success))
