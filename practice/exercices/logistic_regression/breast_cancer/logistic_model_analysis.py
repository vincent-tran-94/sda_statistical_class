import seaborn as sns

if __name__ == "__main__":
    data = sns.load_dataset(name="mpg")
    print(data.info())

    # todo : fitting de la régression logit & analyse des résultats
