from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Data Preparation Pipeline
def create_preprocessor():
    return Pipeline(
        [
            (
                "qualitative_encoder",
                ColumnTransformer(
                    [("onehot", OneHotEncoder(), ["origin"])], remainder="passthrough"
                ),
            ),
            ("nan_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
