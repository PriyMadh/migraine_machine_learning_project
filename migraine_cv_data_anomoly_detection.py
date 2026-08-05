import pandas as pd


# target label: 1=migraine that day; 0=no migraine that day
cv_data = {
    
    #'hard_workout': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    #'processed_food': [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 ,0],
    'avg_temp_during_day': [86, 88, 93, 87, 91, 91, 89, 90, 88, 93, 90, 65, 67, 73, 86, 64, 88, 65, 59, 70, 67, 58],
    
    
    'water_ounces': [10, 32, 25, 20, 10, 30, 40, 40, 35, 18, 64, 64, 128, 132, 100, 54, 80, 90, 60, 70, 140, 132],
    'sun_minutes': [20, 100, 30, 30, 30, 0, 200, 220, 150, 70, 180, 40, 90, 80, 60, 30, 60, 70, 25, 60, 55, 15],
    #'hours_of_sleep': [7.0, 7.0, 6.0, 8.0, 7.5, 7.0, 6.0, 8.0, 8.0, 8.5, 8.5, 7.5, 8.0, 7.2, 8.5, 7.0, 7.8, 8.0, 7.5, 7.0, 8.2, 7.5],
    #'screen_time_hours': [1.0, 0.0, 1.5, 0.5, 0.5, 4.0, 1.5, 1.0, 1.0, 0.0, 0.0, 3.5, 0.0, 1.5, 0.5, 3.0, 0.5, 3.8, 1.5, 0.5, 0.0, 0.0],
    
    # Target Label (11 Migraine days, 11 Non-migeriane days)
    'target_migraine':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(cv_data)
df.to_csv('migraine_anomoly_cv.csv', index=False)
print(f"Dataset compiled! Total tracking days recorded: {len(df)}")
