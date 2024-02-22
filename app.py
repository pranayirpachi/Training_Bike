import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import streamlit.components.v1 as components


class BikeDemandPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.scaler = pickle.load(open(scaler_path, 'rb'))

    def get_datetime(self,date):
        dt = datetime.strptime( date , "%d/%m/%Y")
        return{"day": dt.day , "Month": dt.month, "year": dt.year,"week_day": dt.strftime("%A")}

    def season_to_df(self, seasons):
        seasons_col = ['Spring', 'Summer', 'Winter']
        seasons_data = np.zeros((1, len(seasons_col)), dtype='int')

        df_seasons = pd.DataFrame((seasons_data), columns=seasons_col)

        if seasons in seasons_col:
            df_seasons[seasons] = 1
        return df_seasons

    def days_df(self, week_day):
        days_names = ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
        days_name_data = np.zeros((1, len(days_names)), dtype="int")

        day_df = pd.DataFrame((days_name_data), columns=days_names)

        if week_day in days_names:
            day_df[week_day] = 1

        return day_df

    def predict_demand(self, u_input):
        str_to_date = self.get_datetime(u_input['Date'])

        u_input_list = [u_input['Hours'], u_input['Temperature(°C)'], u_input['Humidity(%)'], u_input['Wind speed (m/s)'],
                        u_input['Visibility (10m)'], u_input['Solar Radiation (MJ/m2)'], u_input['Rainfall(mm)'],
                        u_input['Snowfall (cm)'], u_input['Holiday'], u_input['Functioning Day'], str_to_date['day'],
                        str_to_date['Month'], str_to_date['year']]

        feature_name = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                        'Solar Radiation (MJ/m2)',
                        'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day', 'Month', 'Year']

        df_u_input = pd.DataFrame([u_input_list], columns=feature_name)

        season_df = self.season_to_df(u_input['Season'])

        days_dframe = self.days_df(str_to_date['week_day'])

        final_df = pd.concat([df_u_input, season_df, days_dframe], axis=1)

        sc_data_pred = self.scaler.transform(final_df)
        result = int(np.round(self.model.predict(sc_data_pred)))

        return result


# Main Streamlit app
def main():
    st.title("Bike Demand Prediction App")
    st.write("Enter the following information to predict the bike demand:")

    # Map the values of Holiday and Functioning Day to 0 or 1
    holiday_mapping = {"No Holiday": 0, "Holiday": 1}
    functioning_day_mapping = {"No": 0, "Yes": 1}

    user_input = {
        "Date": st.text_input("Date (dd/mm/yyyy)"),
        "Hours": st.slider("Hours (0-23)", 0, 23),
        "Temperature(°C)": st.number_input("Temperature in celcius"),
        "Humidity(%)": st.number_input("Humidity"),
        "Wind speed (m/s)": st.number_input("Wind Speed"),
        "Visibility (10m)": st.number_input("Visibility"),
        "Solar Radiation (MJ/m2)": st.number_input("Solar Radiation"),
        "Rainfall(mm)": st.number_input("Rainfall"),
        "Snowfall (cm)": st.number_input("SnowFall"),
        "Season": st.selectbox("Season", ["Spring", "Summer", "Winter"]),
        "Holiday": holiday_mapping[st.selectbox("Holiday", ["No Holiday", "Holiday"])],
        "Functioning Day": functioning_day_mapping[st.selectbox("Working day", ["No", "Yes"])]
    }

    if st.button("Predict Demand"):
        model_path = r'D:\Training_Project\rf_model.pkl'
        scaler_path = r'D:\Training_Project\sc_model.pkl'

        predictor = BikeDemandPredictor(model_path, scaler_path)
        prediction = predictor.predict_demand(user_input)

        st.success(f"Predicted Rented Bike Count: {int(round(prediction))}")

if __name__ == "__main__":
    main()
