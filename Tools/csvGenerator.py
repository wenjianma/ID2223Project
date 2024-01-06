import yfinance as yf
import datetime
import pandas as pd

# startDate, as per our convenience we can modify
# startDate = datetime.datetime(2023, 12, 26)


def generate_csv(symbol):
    today = datetime.datetime.now()

    # Symbol on Yahoo Finance
    target_symbol = symbol

    # Download historical data
    target_data = yf.download(target_symbol, start=None, end=today)

    # Check if the DataFrame is not empty
    if not target_data.empty:
        # Reverse the DataFrame to have the latest date first
        # gold_data = gold_data[::-1]

        # Reset the index and create a new 'Index' column
        target_data.reset_index(inplace=True)
        # target_data['Index'] = target_data.index

        # Create a new DataFrame with only 'Index' and 'Close' columns
        target_df = target_data[['Date', 'Close']].copy()

        csv_file_name = f'{symbol}.csv'

        # Save the DataFrame to a CSV file
        target_df.to_csv(csv_file_name, index=False)

        df = pd.read_csv(csv_file_name)

        # Display the DataFrame
        print(df)
    else:
        print("No data available for the specified date.")


generate_csv("GC=F")
generate_csv("^IXIC")
generate_csv("BTC-USD")
