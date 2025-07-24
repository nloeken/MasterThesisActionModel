from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, le_action):
    y_pred = model.predict(X_test)
    print("======= Event Prediction =======")
    print(classification_report(y_test, y_pred, target_names=le_action.classes_))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
