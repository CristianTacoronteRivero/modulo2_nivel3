from nivel3.pipeline_diabetes import Pipeline_diabetes
import pandas as pd

def test_len_features_and_target():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    )
    cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigree",
        "Age",
        "Outcome",
    ]
    diabetes = Pipeline_diabetes(dataframe=df, cols=cols, seed=42)

    X, y = diabetes.select_features(target="Outcome")

    assert X.shape[1] == df.shape[1] - 1
    assert len(y.shape) == 1