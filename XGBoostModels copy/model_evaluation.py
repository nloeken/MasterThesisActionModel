from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# function to evaluate all three models
def evaluate_models(model_action, model_success, model_zone,
                   X_model1_test, X_model2_test, X_model3_test,
                   y_model1_test, y_model2_test, y_model3_test,
                   le_action, df):

    # model 1: next action category
    y_pred_action = model_action.predict(X_model1_test)
    df.loc[X_model1_test.index, "pred_next_action_cat_enc"] = y_pred_action
    df.loc[X_model1_test.index, "pred_next_action_cat"] = le_action.inverse_transform(y_pred_action)

    # model 1: evaluation metrics
    print("===== Model 1: Next Action Category =====")
    print("Accuracy:", accuracy_score(y_model1_test, y_pred_action))
    print("F1 Score (weighted):", f1_score(y_model1_test, y_pred_action, average='weighted'))
    print(classification_report(y_model1_test, y_pred_action, target_names=le_action.classes_))

    # model 1: confusion matrix
    cm1 = confusion_matrix(y_model1_test, y_pred_action)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_action.classes_, yticklabels=le_action.classes_)
    plt.title("Confusion Matrix – Next Action Category")
    plt.xlabel("Predicted event")
    plt.ylabel("Actual event")
    plt.show()

    # model 2: success probability
    y_pred_success = model_success.predict(X_model2_test)
    df.loc[X_model2_test.index, "pred_next_action_success"] = y_pred_success

    # model 2: evaluation metrics
    print("\n===== Model 2: Success Probability =====")
    print("Accuracy:", accuracy_score(y_model2_test, y_pred_success))
    print("F1 Score (weighted):", f1_score(y_model2_test, y_pred_success, average='weighted'))
    print(classification_report(y_model2_test, y_pred_success))

    # model 2: confusion matrix
    cm2 = confusion_matrix(y_model2_test, y_pred_success)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges")
    plt.title("Confusion Matrix – Success Probability")
    plt.xlabel("Predicted success")
    plt.ylabel("Actual success")
    plt.show()

    # model 3: next zone
    y_pred_zone = model_zone.predict(X_model3_test)
    df.loc[X_model3_test.index, "pred_next_zone"] = y_pred_zone

    # model 3: evaluation metrics
    print("\n===== Model 3: Next Zone =====")
    print("Accuracy:", accuracy_score(y_model3_test, y_pred_zone))
    print("F1 Score (weighted):", f1_score(y_model3_test, y_pred_zone, average='weighted'))
    print(classification_report(y_model3_test, y_pred_zone))

    # model 3: confusion matrix
    cm3 = confusion_matrix(y_model3_test, y_pred_zone)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm3, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix – Next Zone (20 Zonen)")
    plt.xlabel("Predicted zone")
    plt.ylabel("Actual zone")
    plt.show()

    return df

# function for SHAP analysis of all three models
def explain_models(model1, model2, model3,
                   X_model1_test, X_model2_test, X_model3_test):

    cols_model1 = model1.get_booster().feature_names
    X_model1_test_aligned = X_model1_test[cols_model1]

    cols_model2 = model2.get_booster().feature_names
    X_model2_test_aligned = X_model2_test[cols_model2]
    
    cols_model3 = model3.get_booster().feature_names
    X_model3_test_aligned = X_model3_test[cols_model3]

    # model 1: SHAP analysis
    print("\n===== SHAP Analysis: Model 1 (Next Action Category) =====")
    explainer1 = shap.TreeExplainer(model1)
    # Nutze jetzt den angepassten DataFrame
    shap_values1 = explainer1.shap_values(X_model1_test_aligned)

    plt.title("Feature Importance – Model 1 (mean SHAP values)")
    shap.summary_plot(shap_values1, X_model1_test_aligned, plot_type="bar", show=False)
    plt.show()

    shap.summary_plot(shap_values1, X_model1_test_aligned, show=False)
    plt.show()

    # model 2: SHAP analysis
    print("\n===== SHAP Analysis: Model 2 (Success Probability) =====")
    explainer2 = shap.TreeExplainer(model2)
    # Nutze jetzt den angepassten DataFrame
    shap_values2 = explainer2.shap_values(X_model2_test_aligned)

    plt.title("Feature Importance – Model 2 (mean SHAP values)")
    shap.summary_plot(shap_values2, X_model2_test_aligned, plot_type="bar", show=False)
    plt.show()

    shap.summary_plot(shap_values2, X_model2_test_aligned, show=False)
    plt.show()

    # model 3: SHAP analysis
    print("\n===== SHAP Analysis: Model 3 (Next Zone) =====")
    explainer3 = shap.TreeExplainer(model3)
    # Nutze jetzt den angepassten DataFrame
    shap_values3 = explainer3.shap_values(X_model3_test_aligned)

    plt.title("Feature Importance – Model 3 (mean SHAP values)")
    shap.summary_plot(shap_values3, X_model3_test_aligned, plot_type="bar", show=False)
    plt.show()

    # HIER trat der Fehler auf, jetzt mit dem angepassten DataFrame
    shap.summary_plot(shap_values3, X_model3_test_aligned, show=False)
    plt.show()