import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_csv='preprocessed_events.csv'):
    df = pd.read_csv(input_csv)
    
    # Optional: nach Match Datum sortieren
    df = df.sort_values('match_id')
    
    train, temp = train_test_split(df, test_size=0.4, shuffle=False)
    valid, test = train_test_split(temp, test_size=0.5, shuffle=False)
    
    train.to_csv('train.csv', index=False)
    valid.to_csv('valid.csv', index=False)
    test.to_csv('test.csv', index=False)
    
    print("Train/Valid/Test split done.")
    return train, valid, test

def main():
    split_data()

if __name__ == "__main__":
    main()
