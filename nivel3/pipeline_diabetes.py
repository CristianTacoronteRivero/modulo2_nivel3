import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA


class Pipeline_diabetes:
    def __init__(self, dataframe: pd.DataFrame, cols: list, seed: int):
        self.dataframe = dataframe
        self.cols = cols
        self.seed = seed

    def select_features(self, target: str) -> pd.DataFrame:
        """Funcion encargada de seleccionar los features y targets de un dataframe

        :param target: nombre del target
        :type target: str
        :return: DataFrame separado en features y target
        :rtype: pd.DataFrame
        """
        self.dataframe.columns = self.cols
        X = self.dataframe.drop(columns=target)
        y = self.dataframe.loc[:, target]
        return X, y

    def split_dataframe(self, X: pd.DataFrame, y: pd.Series):
        """Funcion encargada de seleccionar que datos se van a emplear para entrenar
        un modelo y testearlo

        :param X: DataFrame que contiene los features
        :type X: pd.DataFrame
        :param y: Serie que contiene el target
        :type y: pd.Series
        :return: Conjunto de datos que contiene los features y el target para entrenamiento y test
        :rtype: pd.DataFrame | pd.Series
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.seed
        )
        return X_train, X_test, y_train, y_test

    def fit_model(
        self, X: pd.DataFrame, y: pd.Series, pca: bool = False, params_pca: dict = None
    ):
        """Funcion encargada de entrenar un pipeline cuyo clasificador es del tiempo RandomForest

        :param X: Features de entrenamiento
        :type X: pd.DataFrame
        :param y: Target de entrenamiento
        :type y: pd.Series
        :param pca: especifica si se quiere añadir el proceso PCA o no al pipeline, defaults to False
        :type pca: bool, optional
        :param params_pca: Especifica los parametros del pca, defaults to None
        :type params_pca: dict, optional
        :return: Devuelve un pipeline ya listo para realizar predicciones
        :rtype: Pipeline sklearn
        """
        pipe_num = make_pipeline(StandardScaler(), KNNImputer())
        if pca:
            pipe_num.steps.append(("pca", PCA(params_pca)))
        ct = ColumnTransformer(
            [("pipe_num", pipe_num, list(X.select_dtypes("number").columns))]
        )

        return make_pipeline(ct, RandomForestClassifier()).fit(X, y)

    def export_pipeline(self, name: str, pipeline, mode="wb"):
        """Funcion encargada de exportar en un pickle un modelo Sklearn

        :param name: Nombre del pickle que se quiere generar
        :type name: str
        :param pipeline: Pipeline sklearn
        :type pipeline: Pipeline sklearn
        :param mode: Modo de escritura, defaults to "wb"
        :type mode: str, optional
        """
        with open(f"{name}.pkl", mode) as f:
            pickle.dump(pipeline, f)


if __name__ == "__main__":
    from nivel3.pipeline_diabetes import Pipeline_diabetes

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

    X_train, X_test, y_train, y_test = diabetes.split_dataframe(X, y)

    pipeline = diabetes.fit_model(X_train, y_train, pca=True)

    diabetes.export_pipeline("model", pipeline)
