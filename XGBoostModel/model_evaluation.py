from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt
from XGBoostModel.utils import plot_shap_classwise, plot_feature_correlations, plot_xgb_importance

def evaluate_model(model_action, model_success, X_test, y_action_test, y_success_test, le_action):
    print("===== Prediction: next event =====")
    y_action_pred = model_action.predict(X_test)
    print(classification_report(y_action_test, y_action_pred, target_names=le_action.classes_))

    print("===== Prediction: success of event =====")
    y_success_pred = model_success.predict(X_test)
    print(classification_report(y_success_test, y_success_pred))
    print(f"ROC-AUC: {roc_auc_score(y_success_test, model_success.predict_proba(X_test)[:, 1]):.3f}")

    # SHAP-analysis
    explainer_action = shap.TreeExplainer(model_action)
    shap_values_action = explainer_action.shap_values(X_test)
    shap.summary_plot(shap_values_action, X_test, feature_names=X_test.columns)

    # SHAP per class
    plot_shap_classwise(model_action, X_test, le_action)

    # Feature correlations
    plot_feature_correlations(X_test, X_test.columns)

    # Feature importance
    plot_xgb_importance(model_action, X_test.columns)