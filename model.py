import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import os
import joblib


class Model:
    def __init__(
        self,
        model_name,
        num_months,
        scaler_path=None,
        model_file_path=None,
        num_deps=10,
    ):
        self.model_name = model_name
        self.num_deps = 10
        self.months = num_months
        self.scaler_type = "minmax" if model_name == "naive_bayes" else "standard"

        try:
            scaler_path = scaler_path or os.path.join(
                    "models",
                    "scalers",
                    f"{self.scaler_type}.scaler.{num_months}months.joblib",
                )

            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(e)
            raise Exception("Failed to load scaler")
        model_path = model_file_path or os.path.join(
            "models", model_name, f"{model_name}-{num_months}months.model.joblib"
        )
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(e)
            raise Exception("Failed to load model")

    def preprocess_predict(self, input_df):
        """
        Makes a prediction for a single input row (as a DataFrame).
        Args:
            input_df (pd.DataFrame): DataFrame with a single row of features (raw, not preprocessed).
        Returns:
            int or array: Predicted class label for the input row.
        """
        # Clean and preprocess features
        features = self.preprocess_df(input_df)
        features = self.clean_features(input_df)
        features = self.encode_and_add_months_to_df(features)
        features = self.batch_drop(features, "ave_num_dependencies")
        features = self.batch_drop(features, "ave_number_days")
        features = features.sort_index(
            axis=1, key=lambda col_names: col_names.map(lambda x: str(x)[-1])
        )
        # Scale features
        features = self.scaler.transform(features[self.scaler.feature_names_in_])
        # Predict
        return self.model.predict(features)

    def preprocess_df(self, data_df):
        """
        Preprocesses input DataFrame for model training.

        Args:
            data_df (pd.DataFrame): Raw input data.

        Returns:
            tuple: (features, target) DataFrames.
        """
        # inf_days = self.months * 30 * 1.5
        # data_df = data_df[data_df["speed0"] > 0].reset_index(drop=True)
        data_df = self.batch_clip_to_zero(data_df, "time_diff")
        data_df = data_df.drop(["speed"], axis=1)
        data_df["categorical_target"] = (
            data_df["total_num_cves_in_period"] > 0
        ).astype(int)
        data_df = data_df.drop(["total_num_cves_in_period"], axis=1)
        target = data_df[["categorical_target"]]
        features = data_df.drop(["categorical_target"], axis=1)
        return features, target

    def batch_clip_to_zero(self, df, column_name):
        """Clips values in columns '{column_name}i' (i=0..num_deps-1) to a minimum of zero."""

        for i in range(self.num_deps):
            col_name = f"{column_name}{i}"
            if col_name in df.columns:
                df[col_name] = df[col_name].clip(lower=0)
        return df

    def batch_drop(self, df, col_name):
        """Drops columns named f"{col_name}{i}" for i in range(self.num_deps) from the DataFrame."""

        col_names = [f"{col_name}{i}" for i in range(self.num_deps)]
        df = df.copy()
        for iter_col_name in col_names:
            if iter_col_name in df.columns:
                df = df.drop([iter_col_name], axis=1)
        return df

    def encode_and_add_months_to_df(self, df):
        """
        Encodes month columns as cyclical sine and cosine features, then drops the originals.
        """

        month_mapping = {
            "January": 1,"February": 2,"March": 3,
            "April": 4,"May": 5,"June": 6,
            "July": 7,"August": 8,"September": 9,
            "October": 10,"November": 11,"December": 12,
        }
        todrop = []
        curr_columns = list(df.columns)
        for column in curr_columns:
            if (
                str(column).startswith("dep_release_month")
                or str(column) == "package_release_month"
            ):
                df[str(column) + "_sin"] = (
                    df[column]
                    .map(month_mapping)
                    .apply(
                        lambda x: (
                            np.sin(2 * np.pi * x / 12)
                            if x in month_mapping.values()
                            else 0
                        )
                    )
                )
                df[str(column) + "_cos"] = (
                    df[column]
                    .map(month_mapping)
                    .apply(
                        lambda x: (
                            np.cos(2 * np.pi * x / 12)
                            if x in month_mapping.values()
                            else 0
                        )
                    )
                )
                todrop.append(column)
        df.drop(columns=todrop, inplace=True)
        return df

    def batch_replace(self, df, column, value_to_replace, new_value):
        """
        Replaces specified values in multiple columns of a DataFrame with a new value.
        Parameters:
            df (pandas.DataFrame): The DataFrame containing the columns to modify.
            column (str): The base name of the columns to perform replacements on. Columns are expected to be named as f"{column}{i}".
            value_to_replace: The value to search for and replace in the specified columns.
            new_value: The value to replace value_to_replace with in the specified columns.
        Returns:
            pandas.DataFrame: The DataFrame with the specified values replaced in the target columns.
        """
        for i in range(self.num_deps):
            df[f"{column}{i}"] = df[f"{column}{i}"].replace(value_to_replace, new_value)
        return df

    def compute_and_replace_redundant_columns(self, X):
        """
        Computes and replaces redundant columns in the DataFrame for each dependency.
        For each dependency, calculates:
            - Average CVEs per prior version
            - Average days till first CVE per prior version
            - Fraction of versions with CVEs
        Drops the original columns used for these calculations.
        Args:
            X (pd.DataFrame): Input DataFrame with dependency columns.
        Returns:
            pd.DataFrame: DataFrame with redundant columns replaced.
        """
        X = X.copy()
        ave_cve = "ave_number_cves"
        ave_days = "ave_number_days"
        num_with_cve = "num_with_cves"
        sum_days = "sum_days_till_first_cve"
        total_num_cve = "total_number_cves"
        prior_versions = "prior_versions"
        frac_with_cve = "frac_with_cve"
        for i in range(self.num_deps):
            X[ave_cve + str(i)] = X[total_num_cve + str(i)] / (
                X[prior_versions + str(i)] + 1
            )
            X[ave_days + str(i)] = X[sum_days + str(i)] / (
                X[prior_versions + str(i)] + 1
            )
            X[frac_with_cve + str(i)] = X[num_with_cve + str(i)] / (
                X[prior_versions + str(i)] + 1
            )
            X.drop(
                [sum_days + str(i), total_num_cve + str(i), num_with_cve + str(i)],
                axis=1,
                inplace=True,
            )
        return X

    def clean_features(self, features_df):
        """Cleans and preprocesses the input features DataFrame."""
        df = self.batch_drop(features_df, "ave_number_dependencies")
        df = self.batch_drop(df, "dep_release_month")
        features = self.encode_and_add_months_to_df(df)
        features = self.batch_replace(features, "prior_versions", -1, 0)
        X = self.compute_and_replace_redundant_columns(features)
        X = self.batch_drop(X, "prior_versions")
        X = self.batch_drop(X, "num_with_cves")
        return X

    def create_interpolated_df(self, original):
        """
        Interpolates dependency-related features for each row in the input DataFrame.
        Args:
            original (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with interpolated dependency features.
        """

        _dep_features = [
            "speed",
            "time_diff",
            "dep_release_month",
            "prior_versions",
            "num_with_cves",
            "total_number_cves",
            "sum_days_till_first_cve",
            "ave_num_dependencies",
        ]
        new_columns = []
        for x in range(self.num_deps):
            for feature in _dep_features:
                new_columns.append(f"{feature}{x}")
        interpolated_data = [
            self.repeat_fill(row, self.num_deps) for _, row in original.iterrows()
        ]
        new_df = pd.DataFrame(interpolated_data, columns=new_columns)
        return new_df

    def repeat_fill(self, row, num_columns):
        """
        Evenly repeats available dependency indices to fill missing dependency features in a row.
        Args:
            row (dict or pd.Series): Row with dependency features named as "<feature><index>".
            num_columns (int): Number of dependency columns.
        Returns:
            list: Feature values for all dependencies, filling missing by repeating available ones.
        """

        _dep_features = [
            "speed",
            "time_diff",
            "dep_release_month",
            "prior_versions",
            "num_with_cves",
            "total_number_cves",
            "sum_days_till_first_cve",
            "ave_num_dependencies",
        ]
        available = [
            dep_idx
            for dep_idx in range(num_columns)
            if row["speed" + str(dep_idx)] != -1
        ]
        if not available:
            available = [0]
        each_size = [num_columns // len(available) for _ in range(num_columns)]
        remainder = num_columns % len(available)
        for idx in range(len(each_size)):
            if remainder < 1:
                break
            each_size[idx] += 1
            remainder -= 1
        interpolated = []
        for idx, x in enumerate(available):
            interpolated.extend([x] * each_size[idx])
        new_features = []
        for dep in range(num_columns):
            for feat in _dep_features:
                new_features.append(row[f"{feat}{interpolated[dep]}"])
        return new_features

    def predict(self, X_test):
        """Predict target values for the given test data."""
        return self.model.predict(X_test)


if __name__ == "__main__":
    sample_row = pd.read_csv("sample_row.csv")
    for model_name in ["random_forest", "xgboost", "logreg", "naive_bayes"]:
        for window in [3, 6, 12]:
            model = Model(model_name=model_name, num_months=window)
            prediction = model.preprocess_predict(sample_row)
            print(model_name, window, prediction)
