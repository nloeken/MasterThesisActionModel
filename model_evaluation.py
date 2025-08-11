from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
from utils import plot_shap_classwise, plot_feature_correlations, plot_xgb_importance

def evaluate_model(model, X_test, y_test, le_action):
    print("======= Event Prediction =======")
    print(classification_report(y_test, model.predict(X_test), target_names=le_action.classes_))

    # Global SHAP Summary
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

    # Feature importance for every action class
    plot_shap_classwise(model, X_test, le_action)

    # correlations between features
    plot_feature_correlations(X_test, X_test.columns)

    # XGBoost feature importance
    plot_xgb_importance(model, X_test.columns)