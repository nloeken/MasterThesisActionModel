import numpy as np
import pandas as pd

def compute_expected_threat(df, model_zone, xt_values, features):
    """
    Berechnet den erwarteten xT-Gewinn pro Aktion basierend auf:
    - Zone-Vorhersagen (model_zone)
    - xT-Wert pro Zone (xt_values)
    """
    X = df[features].fillna(0)
    zone_probs = model_zone.predict_proba(X)
    expected_xt_gain = []

    for i, probs in enumerate(zone_probs):
        cur_zone = df.iloc[i]['current_zone']  # aktuelle Zone der Aktion
        cur_xt = xt_values.get(cur_zone, 0.0)
        next_xt = sum(probs[z] * xt_values.get(z, 0.0) for z in range(len(probs)))
        xt_gain = next_xt - cur_xt
        expected_xt_gain.append(xt_gain)

    return np.array(expected_xt_gain)

def main():
    # --- Daten laden ---
    df = pd.read_csv("match_data.csv")  # Enthält aktuelle Aktionen + Features

    # --- xT-Werte für 20 Zonen ---
    xt_values = {i: val for i, val in enumerate([
        0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.10,
        0.11, 0.12, 0.13, 0.14, 0.15,
        0.16, 0.17, 0.18, 0.19, 0.20
    ])}

    # --- Feature-Spalten für die Modelle ---
    base_features = [
        'x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box', 'in_cross_zone',
        'nearby_opponents', 'high_pressure', 'low_pressure', 'orientation',
        'free_teammates', 'is_late_game', 'is_losing', 'duration', 'prev_event_success'
    ]
    zone_features = [col for col in df.columns if col in base_features]
    action_features = zone_features + [col for col in df.columns if col.startswith((
        'cur_act_', 'prev_act_', 'team_', 'pos_', 'phase_', 'half_'
    ))]

    # --- Modelle laden oder trainieren ---
    # model1: Next Action Prediction
    # model3: Next Zone Prediction
    # Annahme: Modelle sind bereits trainiert
    # best_model1, best_model3 = train_models(...)[0], train_models(...)[2]

    # --- Vorhersagen ---
    X_zone = df[zone_features].fillna(0)
    X_action = df[action_features].fillna(0)

    df['predicted_zone'] = model3.predict(X_zone)
    df['predicted_action'] = model1.predict(X_action)

    # --- Expected Threat berechnen ---
    df['expected_xt_gain'] = compute_expected_threat(df, model3, xt_values, zone_features)

    # Optional: xT pro vorhergesagter Aktion analysieren
    action_xt_summary = df.groupby('predicted_action')['expected_xt_gain'].mean().sort_values(ascending=False)
    print("\nExpected Threat pro vorhergesagter Aktion:")
    print(action_xt_summary)

    # --- Speichern ---
    df.to_csv("match_data_with_actions_xT.csv", index=False)
    print("Daten mit predicted_action, predicted_zone und expected_xt_gain gespeichert.")

if __name__ == "__main__":
    main()
