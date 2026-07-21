import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="TurbineGuard", page_icon="✈️", layout="centered")

API_URL = "https://turbineguard-api.onrender.com/predict"

SAMPLES = {
    "Healthy Engine": {
        "time_cycles": 30.0, "sensor_2": 642.24, "sensor_3": 1574.15, "sensor_4": 1396.49,
        "sensor_7": 555.14, "sensor_8": 2388.0, "sensor_9": 9054.65, "sensor_11": 47.06,
        "sensor_12": 523.06, "sensor_13": 2388.03, "sensor_14": 8143.79, "sensor_15": 8.3799,
        "sensor_17": 392.0, "sensor_20": 39.27, "sensor_21": 23.3407,
        "sensor_9_rollavg": 9055.854, "sensor_2_rollavg": 642.036, "sensor_3_rollavg": 1583.358,
        "sensor_4_rollavg": 1396.666, "sensor_7_rollavg": 554.926, "sensor_8_rollavg": 2388.016,
        "sensor_11_rollavg": 47.132, "sensor_12_rollavg": 522.708, "sensor_13_rollavg": 2388.002,
        "sensor_14_rollavg": 8141.766, "sensor_15_rollavg": 8.38804, "sensor_17_rollavg": 391.4,
        "sensor_20_rollavg": 39.07, "sensor_21_rollavg": 23.36984,
        "sensor_2_slope": -0.014, "sensor_3_slope": -2.183, "sensor_4_slope": -1.942,
        "sensor_7_slope": 0.076, "sensor_8_slope": -0.011, "sensor_9_slope": -0.515,
        "sensor_11_slope": 0.024, "sensor_12_slope": 0.123, "sensor_13_slope": 0.006,
        "sensor_14_slope": 0.669, "sensor_15_slope": -0.00701, "sensor_17_slope": 0.2,
        "sensor_20_slope": 0.017, "sensor_21_slope": -0.00721
    },
    "Moderate Engine": {
        "time_cycles": 157.0, "sensor_2": 642.09, "sensor_3": 1590.38, "sensor_4": 1403.98,
        "sensor_7": 553.84, "sensor_8": 2388.06, "sensor_9": 9068.17, "sensor_11": 47.29,
        "sensor_12": 522.17, "sensor_13": 2388.08, "sensor_14": 8145.02, "sensor_15": 8.4108,
        "sensor_17": 392.0, "sensor_20": 38.8, "sensor_21": 23.337,
        "sensor_9_rollavg": 9062.834, "sensor_2_rollavg": 642.532, "sensor_3_rollavg": 1588.942,
        "sensor_4_rollavg": 1404.748, "sensor_7_rollavg": 553.936, "sensor_8_rollavg": 2388.042,
        "sensor_11_rollavg": 47.36, "sensor_12_rollavg": 521.926, "sensor_13_rollavg": 2388.066,
        "sensor_14_rollavg": 8139.806, "sensor_15_rollavg": 8.42498, "sensor_17_rollavg": 392.4,
        "sensor_20_rollavg": 38.842, "sensor_21_rollavg": 23.32718,
        "sensor_2_slope": -0.189, "sensor_3_slope": -0.017, "sensor_4_slope": -0.037,
        "sensor_7_slope": 0.018, "sensor_8_slope": 0.006, "sensor_9_slope": 2.12,
        "sensor_11_slope": -0.022, "sensor_12_slope": -0.008, "sensor_13_slope": 0.004,
        "sensor_14_slope": 1.824, "sensor_15_slope": -0.00957, "sensor_17_slope": 0.0,
        "sensor_20_slope": 0.017, "sensor_21_slope": -0.00885
    },
    "Critical Engine": {
        "time_cycles": 188.0, "sensor_2": 643.64, "sensor_3": 1603.1, "sensor_4": 1434.88,
        "sensor_7": 551.23, "sensor_8": 2388.28, "sensor_9": 9050.97, "sensor_11": 48.17,
        "sensor_12": 519.61, "sensor_13": 2388.3, "sensor_14": 8124.14, "sensor_15": 8.4977,
        "sensor_17": 396.0, "sensor_20": 38.61, "sensor_21": 22.9752,
        "sensor_9_rollavg": 9053.574, "sensor_2_rollavg": 643.57, "sensor_3_rollavg": 1599.962,
        "sensor_4_rollavg": 1429.986, "sensor_7_rollavg": 551.22, "sensor_8_rollavg": 2388.286,
        "sensor_11_rollavg": 48.178, "sensor_12_rollavg": 519.572, "sensor_13_rollavg": 2388.276,
        "sensor_14_rollavg": 8128.09, "sensor_15_rollavg": 8.52144, "sensor_17_rollavg": 396.2,
        "sensor_20_rollavg": 38.498, "sensor_21_rollavg": 23.05204,
        "sensor_2_slope": 0.179, "sensor_3_slope": -0.152, "sensor_4_slope": 2.486,
        "sensor_7_slope": -0.013, "sensor_8_slope": 0.0, "sensor_9_slope": -0.061,
        "sensor_11_slope": 0.034, "sensor_12_slope": -0.118, "sensor_13_slope": 0.014,
        "sensor_14_slope": -2.518, "sensor_15_slope": 0.00139, "sensor_17_slope": 0.1,
        "sensor_20_slope": 0.016, "sensor_21_slope": -0.00875
    }
}

st.title("✈️ TurbineGuard")
st.caption("Aircraft engine Remaining Useful Life (RUL) prediction — tuned XGBoost, live via FastAPI")

st.markdown(
    "Airlines maintain engines on fixed schedules, regardless of actual condition. "
    "This model predicts how many flight cycles an engine has left, from live sensor "
    "readings — enabling maintenance based on real condition, not a calendar guess."
)

engine_choice = st.selectbox("Choose a sample engine", list(SAMPLES.keys()))

if st.button("Predict RUL", type="primary"):
    with st.spinner("Calling the live model — first request after inactivity may take up to a minute..."):
        try:
            response = requests.post(
                API_URL, json={"features": SAMPLES[engine_choice]}, timeout=90
            )
            response.raise_for_status()
            result = response.json()
            rul = result["predicted_RUL"]
            risk = result["risk_tier"]

            color_map = {"Critical": "#c0392b", "Warning": "#e67e22",
                         "Moderate": "#2980b9", "Healthy": "#27ae60"}
            color = color_map.get(risk, "#7f8c8d")

            col1, col2 = st.columns(2)
            col1.metric("Predicted RUL", f"{rul} cycles")
            col2.markdown(
                f"<h3 style='color:{color};'>Risk Tier: {risk}</h3>",
                unsafe_allow_html=True
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rul,
                gauge={
                    'axis': {'range': [0, 125]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': "#fde0e0"},
                        {'range': [20, 50], 'color': "#fdf0d5"},
                        {'range': [50, 90], 'color': "#e0ecfd"},
                        {'range': [90, 125], 'color': "#e0fde3"}
                    ]
                },
                title={'text': "Remaining Useful Life (cycles)"}
            ))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with st.expander("View raw sensor inputs sent to the model"):
    st.json(SAMPLES[engine_choice])

st.markdown("---")
st.caption(
    "Model: Tuned XGBoost, test MAE 12.24 cycles · Trained on NASA C-MAPSS FD001 · "
    "[API docs](https://turbineguard-api.onrender.com/docs) · "
    "[GitHub](https://github.com/gaganasindhu-ml/turbineguard-api)"
)
