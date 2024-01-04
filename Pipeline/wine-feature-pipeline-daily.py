import modal

LOCAL = False
# run code locally or deploy it on Modal

if LOCAL == False:
    stub = modal.Stub("wine_daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    # we can set schedule for re-exection here
    def f():
        g()


def generate_wine(name, volatile_acidity_max, volatile_acidity_min, chlorides_max, chlorides_min, density_max, density_min, alcohol_max, alcohol_min
                  ):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"volatile_acidity": [random.uniform(volatile_acidity_max, volatile_acidity_min)],
                       "chlorides": [random.uniform(chlorides_max, chlorides_min)],
                       "density": [random.uniform(density_max, density_min)],
                       "alcohol": [random.uniform(alcohol_max, alcohol_min)],
                       })
    df['quality'] = name
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    q3_df = generate_wine(
        3, 0.25, 0.75, 0.025, 0.15, 0.992, 1.001, 8, 12.5)
    q4_df = generate_wine(
        4, 0.2, 1.05, 0.02, 0.125, 0.989, 1.001, 8.5, 13.2)
    q5_df = generate_wine(
        5, 0.15, 0.8, 0.01, 0.125, 0.988, 1.002, 8, 13)
    q6_df = generate_wine(
        6, 0.1, 0.65, 0.02, 0.12, 0.987, 1.002, 8.5, 13.5)
    q7_df = generate_wine(
        7, 0.1, 0.6, 0.02, 0.115, 0.987, 1.002, 8.6, 14)
    q8_df = generate_wine(
        8, 0.15, 0.6, 0.02, 0.09, 0.988, 1, 8.6, 14)
    q9_df = generate_wine(
        9, 0.25, 0.35, 0.02, 0.035, 0.989, 0.997, 10.5, 13)

    # randomly pick one of these 7 and write it to the featurestore
    pick_random = random.uniform(0, 100)
    if pick_random >= 98:
        wine_df = q3_df
        print("Q3 wine added")
    elif pick_random >= 93:
        wine_df = q4_df
        print("Q4 wine added")
    elif pick_random >= 61:
        wine_df = q5_df
        print("Q5 wine added")
    elif pick_random >= 24:
        wine_df = q6_df
        print("Q6 wine added")
    elif pick_random >= 6:
        wine_df = q7_df
        print("Q7 wine added")
    elif pick_random >= 1:
        wine_df = q8_df
        print("Q8 wine added")
    else:
        wine_df = q9_df
        print("Q9 wine added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine", version=4)
    # pay attention to version
    wine_fg.insert(wine_df)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        modal.runner.deploy_stub(stub)
        f.remote()
