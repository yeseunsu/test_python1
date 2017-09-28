import tempfile

import os
import urllib

import numpy as np
import tensorflow as tf

import urllib.request

import pandas as pd

# train_file = tempfile.NamedTemporaryFile(mode='w')
# test_file = tempfile.NamedTemporaryFile(mode='w')
#
# urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
# urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)


# Data sets
train_file = "sample_data/adult.data"
train_file_url = "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data"

test_file = "sample_data/adult.test"
test_file_url = "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test"

def main():

  # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(train_file):
        with urllib.request.urlopen(train_file_url) as url:
            raw = url.read()
            with open(train_file, "w") as f:
                f.write(str(raw,'utf-8'))

    if not os.path.exists(test_file):
        with urllib.request.urlopen(test_file_url) as url:
            raw = url.read()
            with open(test_file, "w") as f:
                f.write(str(raw, 'utf-8'))

    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]


    # df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
    # df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
    #
    # LABEL_COLUMN = "label"
    # df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    # df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    #
    # df_data = df_train.dropna(how="any", axis=0)
    # print(df_data.shape)


    def input_fn(data_file, num_epochs, shuffle):
        """Input builder function."""
        df_data = pd.read_csv(
            tf.gfile.Open(data_file),
            names=COLUMNS,
            skipinitialspace=True,
            engine="python",
            skiprows=1)
        # remove NaN elements
        df_data = df_data.dropna(how="any", axis=0)
        labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
        return tf.estimator.inputs.pandas_input_fn(
            x=df_data,
            y=labels,
            batch_size=100,
            num_epochs=num_epochs,
            shuffle=shuffle,
            num_threads=5)


    # Base Categorical Feature Columns
    gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        "education", [
            "Bachelors", "HS-grad", "11th", "Masters", "9th",
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
            "Preschool", "12th"
        ])
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        "marital_status", [
            "Married-civ-spouse", "Divorced", "Married-spouse-absent",
            "Never-married", "Separated", "Married-AF-spouse", "Widowed"
        ])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        "relationship", [
            "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
            "Other-relative"
        ])
    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        "workclass", [
            "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
            "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
        ])
    native_country = tf.feature_column.categorical_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000)

    # Base continuous feature columns
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])


    base_columns = [gender, native_country, education, occupation, workclass, relationship, age_buckets,]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ["education", "occupation"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, "education", "occupation"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ["native_country", "occupation"], hash_bucket_size=1000)
    ]

    model_dir = tempfile.mkdtemp()
    m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns + crossed_columns)

    train_steps=2000
    m.train(input_fn=input_fn(train_file, num_epochs=None, shuffle=True), steps=train_steps)

    results = m.evaluate(input_fn=input_fn(test_file, num_epochs=1, shuffle=False), steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))



if __name__ == "__main__":
    main()