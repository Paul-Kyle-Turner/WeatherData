# WeatherData

Assignment do as much as possbile with the data provided in 10 hours.  Original csv file renamed for privacy to data.csv.

1) first_look.ipynb
3) eda.ipynb
4) modeling.ipynb

Temperature in degree C as label.

Don't go read it, here is the interesting parts :


` 
 Numeric - ['RelativeHumidity', 'Temperature_Celsius', 'Pressure_Millibar', 'MoonInfo_Phase', 'MoonInfo_Illumination', 'MoonInfo_Age', 'Precipitation_Hour_Cm', 'Precipitation_Year_Cm', 'Precipitation_Month_Cm', 'Precipitation_Week_Cm', 'Precipitation_Day_Cm', 'Wind_X', 'Wind_Y', 'Wind_Gust_KilometersPerHour'] 

One Hot Encoding of current conditions and moon data. (Not expecting a signal from the moon data, was going to do a comparison.  I didn't get to it.)
       
`

`
wind_direction_deg = extracted_dataframe_double['Wind_Direction_Degree']
wind_speed_kph = extracted_dataframe_double['Wind_Speed_KilometersPerHour']
wind_direction_rad = wind_direction_deg * np.pi / 180
wind_x_vector = wind_speed_kph * np.cos(wind_direction_rad)
wind_y_vector = wind_speed_kph * np.sin(wind_direction_rad)
`


` 
moving_average = data[numeric_columns].rolling(window=3).mean()
normalized_moving_average_data = (moving_average - moving_average.min()) / (moving_average.max() - moving_average.min())
normalized_moving_average_data = normalized_moving_average_data.dropna()
normalized_moving_average_data.head()
`

Sources used : 
  Used tensorflow website, took the window tool they made.
  window_generator.py
