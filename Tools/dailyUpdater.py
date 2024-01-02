import yfinance as yf
import datetime
import pandas as pd


def update_csv(symbol):
    # today = datetime.datetime.now()
    today = datetime.datetime(2024, 1, 2)

    # Symbol on Yahoo Finance
    target_symbol = symbol

    # Download historical data
    try:
        target_data = yf.download(target_symbol, start=today, end=None)
        target_data.reset_index(inplace=True)
        target_data['Date'] = target_data['Date'].dt.strftime('%Y-%m-%d')
        print(target_data)
    except Exception as e:
        target_data = pd.DataFrame()

    if not target_data.empty:
        csv_file_name = f'{symbol}.csv'
        df = pd.read_csv(csv_file_name)

        # Convert 'Date' column to a string with the desired format
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        # Check if the last row's 'Date' in the existing DataFrame matches the first row's 'Date' in target_data
        if not df.empty and df['Date'].iloc[-1] == target_data['Date'].iloc[0]:
            print(
                "Data for the specified date already exists in the CSV file. Skipping merge.")
        else:
            # Concatenate the DataFrames
            df_updated = pd.concat([df, target_data])

            # Save the updated DataFrame back to the CSV file
            df_updated.to_csv(csv_file_name, index=False)
            print(f'Data for {symbol} updated successfully.')
    else:
        print("No data available for the specified date.")


update_csv("GC=F")
