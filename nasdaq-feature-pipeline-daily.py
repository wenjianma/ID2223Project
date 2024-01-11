import modal
import yfinance as yf
import datetime
import pandas as pd

LOCAL = False

if LOCAL == False:
    stub = modal.Stub("nasdaq_daily")
    image = modal.Image.debian_slim().pip_install("hopsworks","yfinance")

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()



def get_daily_update():
    # the symbol for nasdaq
    symbol = "GC=F"

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    # Symbol on Yahoo Finance
    target_symbol = symbol

    # Download historical data
    try:
        target_data = yf.download(target_symbol, start=today, end=None)
        target_data.reset_index(inplace=True)
        target_data['Date'] = target_data['Date'].dt.strftime('%Y-%m-%d')
        target_data['Date'] = pd.to_datetime(target_data['Date'])
        target_data = target_data[['Date', 'Close']]
        print(target_data)
    except Exception as e:
        target_data = pd.DataFrame()

    # if not target_data.empty:
    #     csv_file_name = f'{symbol}.csv'
    #     df = pd.read_csv(csv_file_name)
    #
    #     # Convert 'Date' column to a string with the desired format
    #     df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    #
    #     # Check if the last row's 'Date' in the existing DataFrame matches the first row's 'Date' in target_data
    #     if not df.empty and df['Date'].iloc[-1] == target_data['Date'].iloc[0]:
    #         print(
    #             "Data for the specified date already exists in the CSV file. Skipping merge.")
    #     else:
    #         # Concatenate the DataFrames
    #         df_updated = pd.concat([df, target_data])
    #
    #         # Save the updated DataFrame back to the CSV file
    #         df_updated.to_csv(csv_file_name, index=False)
    #         print(f'Data for {symbol} updated successfully.')
    # else:
    #     print("No data available for the specified date.")

    return target_data


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    nasdaq_df = get_daily_update()

    nasdaq_fg = fs.get_feature_group(name="nasdaq")
    nasdaq_fg.insert(nasdaq_df)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        modal.runner.deploy_stub(stub)
        f.remote()
