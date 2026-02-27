from sklearn.linear_model import LinearRegression
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("data_fit6.csv")
    
    # X = data[['score_text', 'score_visual']]
    # filtered_df = data[data['type_data'] == 'iiw'] 
    # filtered_df = data.sample(frac=0.8, random_state=42)
    eighty_percent_rows = int(0.8 * len(data))

    # Select the first 80% of the rows
    df_80_percent = data.iloc[:eighty_percent_rows]
    # filtered_df = df_80_percent
    filtered_df = data
    
    X = filtered_df[['score_text', 'score_visual_1', 'score_visual_2', 'score_visual_3','score_visual_4']]
    y = filtered_df['label']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(model.coef_)
    print(model.intercept_)