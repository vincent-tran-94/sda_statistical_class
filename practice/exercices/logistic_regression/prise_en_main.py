from demos.utlis import setup_plot, get_data, ProblemType, plot_classification

if __name__ == "__main__":

    colors = setup_plot()

    X, y = get_data(type=ProblemType.CLASSIFICATION)
    plot_classification(X=X, y=y, colors=colors)

    # todo : Fit the logit regression on this, and anaylze the results
