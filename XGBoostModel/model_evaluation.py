from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def evaluate_models(model_action, model_success,
                   X_model1_test, X_model2_test, 
                   y_model1_test, y_model2_test, 
                   le_action, df):

    # Model 1: Next action category
    y_pred_action = model_action.predict(X_model1_test)
    df.loc[X_model1_test.index, "pred_next_action_cat_enc"] = y_pred_action
    df.loc[X_model1_test.index, "pred_next_action_cat"] = le_action.inverse_transform(y_pred_action)

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
    plt.xlabel("Predicted event")
    plt.ylabel("Actual event")
    plt.show()

    # Model 2: Success probability
    y_pred_success = model_success.predict(X_model2_test)
    df.loc[X_model2_test.index, "pred_next_action_success"] = y_pred_success

    print("\n=== Model 2: Success Probability ===")
    print("Accuracy:", accuracy_score(y_model2_test, y_pred_success))
    print("F1 Score (weighted):", f1_score(y_model2_test, y_pred_success, average='weighted'))
    print(classification_report(y_model2_test, y_pred_success))

    return df

# SHAP-Analysis
def explain_models(model1, model2, X_model1_test, X_model2_test):
    # background dataset (sample)
    background1 = X_model1_test.sample(200, random_state=42)
    background2 = X_model2_test.sample(200, random_state=42)

    # SHAP explainer
    explainer1 = shap.TreeExplainer(model1)
    explainer2 = shap.TreeExplainer(model2)

    # calculation of shap values
    shap_values1 = explainer1.shap_values(X_model1_test)
    shap_values2 = explainer2.shap_values(X_model2_test)

    # Model 1 (Multiclass)
    plt.title("Feature Importance – Model 1 (mean SHAP values)")
    shap.summary_plot(shap_values1, X_model1_test, plot_type="bar", show=False)
    plt.show()

    # Class-specific overview
    shap.summary_plot(shap_values1, X_model1_test, show=False)
    plt.show()

    # Model 2 (Binary)
    plt.title("Feature Importance – Model 2 (mean SHAP values)")
    shap.summary_plot(shap_values2, X_model2_test, plot_type="bar", show=False)
    plt.show()

    # Class-specific overview
    shap.summary_plot(shap_values2, X_model2_test, show=False)
    plt.show()