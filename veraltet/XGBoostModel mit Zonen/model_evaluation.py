from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def evaluate_models(model_action, model_success, model_zone,
                   X_model1_test, X_model2_test, X_model3_test,
                   y_model1_test, y_model2_test, y_model3_test,
                   le_action, df):
    """
    Evaluation der drei Modelle:
    - Model 1: Next Action Category (multiclass)
    - Model 2: Success Probability (binary)
    - Model 3: Next Zone (multiclass, 20 Klassen)
    """

    # === Model 1: Next Action Category ===
    y_pred_action = model_action.predict(X_model1_test)
    df.loc[X_model1_test.index, "pred_next_action_cat_enc"] = y_pred_action
    df.loc[X_model1_test.index, "pred_next_action_cat"] = le_action.inverse_transform(y_pred_action)

    print("=== Model 1: Next Action Category ===")
    print("Accuracy:", accuracy_score(y_model1_test, y_pred_action))
    print("F1 Score (weighted):", f1_score(y_model1_test, y_pred_action, average='weighted'))
    print(classification_report(y_model1_test, y_pred_action, target_names=le_action.classes_))

    cm1 = confusion_matrix(y_model1_test, y_pred_action)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_action.classes_, yticklabels=le_action.classes_)
    plt.title("Confusion Matrix – Next Action Category")
    plt.xlabel("Predicted event")
    plt.ylabel("Actual event")
    plt.show()

    # === Model 2: Success Probability ===
    y_pred_success = model_success.predict(X_model2_test)
    df.loc[X_model2_test.index, "pred_next_action_success"] = y_pred_success

    print("\n=== Model 2: Success Probability ===")
    print("Accuracy:", accuracy_score(y_model2_test, y_pred_success))
    print("F1 Score (weighted):", f1_score(y_model2_test, y_pred_success, average='weighted'))
    print(classification_report(y_model2_test, y_pred_success))

    cm2 = confusion_matrix(y_model2_test, y_pred_success)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges")
    plt.title("Confusion Matrix – Success Probability")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # === Model 3: Next Zone ===
    y_pred_zone = model_zone.predict(X_model3_test)
    df.loc[X_model3_test.index, "pred_next_zone"] = y_pred_zone

    print("\n=== Model 3: Next Zone ===")
    print("Accuracy:", accuracy_score(y_model3_test, y_pred_zone))
    print("F1 Score (weighted):", f1_score(y_model3_test, y_pred_zone, average='weighted'))
    print(classification_report(y_model3_test, y_pred_zone))

    cm3 = confusion_matrix(y_model3_test, y_pred_zone)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm3, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix – Next Zone (20 Zonen)")
    plt.xlabel("Predicted zone")
    plt.ylabel("Actual zone")
    plt.show()

    return df


# === SHAP-Analyse ===
def explain_models(model1, model2, model3,
                   X_model1_test, X_model2_test, X_model3_test):
    """
    SHAP-Analyse für alle drei Modelle.
    """

    # Model 1
    explainer1 = shap.TreeExplainer(model1)
    shap_values1 = explainer1.shap_values(X_model1_test)

    plt.title("Feature Importance – Model 1 (mean SHAP values)")
    shap.summary_plot(shap_values1, X_model1_test, plot_type="bar", show=False)
    plt.show()

    shap.summary_plot(shap_values1, X_model1_test, show=False)
    plt.show()

    # Model 2
    explainer2 = shap.TreeExplainer(model2)
    shap_values2 = explainer2.shap_values(X_model2_test)

    plt.title("Feature Importance – Model 2 (mean SHAP values)")
    shap.summary_plot(shap_values2, X_model2_test, plot_type="bar", show=False)
    plt.show()

    shap.summary_plot(shap_values2, X_model2_test, show=False)
    plt.show()

    # Model 3
    explainer3 = shap.TreeExplainer(model3)
    shap_values3 = explainer3.shap_values(X_model3_test)

    plt.title("Feature Importance – Model 3 (mean SHAP values)")
    shap.summary_plot(shap_values3, X_model3_test, plot_type="bar", show=False)
    plt.show()

    shap.summary_plot(shap_values3, X_model3_test, show=False)
    plt.show()
