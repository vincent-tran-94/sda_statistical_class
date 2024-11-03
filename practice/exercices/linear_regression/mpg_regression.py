import seaborn

if __name__ == "__main__":
    dataset = seaborn.load_dataset(name="mpg")

    print(dataset.info())
    # fit an OLS to predict the mpg column
    # certain columns are categorical, you need to encode them first before fitting the OLS
