from demos.utlis import setup_plot, get_data, plot_data, ProblemType

if __name__ == "__main__":

    colors = setup_plot()

    X, y = get_data(type=ProblemType.REGRESSION)
    plot_data(X=X, y=y, colors=colors)

    # todo : Fit the OLS on this, and verify the regression hypothesis
