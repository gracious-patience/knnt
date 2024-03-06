# ZOMB VERSION
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
import pandas as pd
import dnnr
import random
random.seed(42)
np.random.seed(42)

# Regression datasets with only numerical features from https://arxiv.org/pdf/2207.08815.pdf 
# (Why do tree-based models still outperform deep learning on tabular data?)
links = [
    {"url": "https://api.openml.org/data/download/22103266/dataset", "name": "Brazilian_houses"},
    {"url": "https://api.openml.org/data/download/22103261/dataset", "name":"wine"},
    {"url": "https://api.openml.org/data/download/22103262/dataset", "name": "ailerons"},
    {"url": "https://api.openml.org/data/download/22103263/dataset", "name": "houses"},
    {"url": "https://api.openml.org/data/download/22103264/dataset", "name": "house_16H"},
    {"url": "https://api.openml.org/data/download/22103267/dataset", "name": "Bike_Sharing_Demand"},
    # {"url": "https://api.openml.org/data/download/22103268/dataset", "name": "nyc-taxi-green-dec-2016"},
    {"url": "https://api.openml.org/data/download/22103269/dataset", "name": "house_sales"},
    {"url": "https://api.openml.org/data/download/22103270/dataset", "name": "sulfur"},
    {"url": "https://api.openml.org/data/download/22103271/dataset", "name": "medical_charges"},
    {"url": "https://api.openml.org/data/download/22103272/dataset", "name": "MiamiHousing2016"},
    {"url": "https://api.openml.org/data/download/22103273/dataset", "name": "superconduct"},
    {"url": "", "name": "cpu"},
    {"url": "", "name": "diamond"},
    {"url": "", "name": "isolet"},
    {"url": "", "name": "pol"},
]

# preprocessing
standard_scaling = StandardScaler()

# kNNt
with open("./results/knnt.csv", "w") as f:
    # first line in csv file
    f.write("dataset,k,ks,order,scaling,metric,mean,std\n")
    # iterate over datasets
    for dataset in links[4:8]:
        # read data
        data = pd.read_csv(f"./datasets/{dataset['name']}")
        print(f"working with {dataset['name']}")
        features = list(data.columns)[:-1]
        labels = list(data.columns)[-1]
        data_X = data[data.columns.intersection(features)]
        data_y = data[data.columns.intersection([labels])]
        X, y = data_X.to_numpy(), data_y.to_numpy().flatten()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
        # as in DNNR paper vary hyperparameters according to the size of dataset
        # small
        if X.shape[0] < 2000:
            ks = [1, 2, 3, 5, 7]
            kks = sorted(random.sample(range(2*X.shape[1], 15*X.shape[1]), 30))
        # medium
        elif X.shape[0] < 50000:
            ks = [3, 4]
            kks = sorted(random.sample(range(2*X.shape[1], 18*X.shape[1]), 20))
        # large
        else:
            ks = [3]
            kks = sorted(random.sample(range(2*X.shape[1], 12*X.shape[1]), 14))

        print(f"der neighbors: {kks}")
        for k in ks:
            for kk in kks:
                for order in ["1","2"]:
                    for scaling in ["learned", "no_scaling"]:
                        for metric in ["r2"]:
                            model = dnnr.DNNR(n_neighbors=k, n_derivative_neighbors=kk, order=order, solver="scipy_lsqr", scaling=scaling)
                            reg = make_pipeline(standard_scaling, model)
                            scores = cross_val_score(reg, X_train, y_train, scoring=metric, cv=10, n_jobs=1)  
                            mean, std = scores.mean(), scores.std()
                            print(f"{dataset['name']},{k},{kk},{order},{scaling},{metric},{mean},{std}")
                            f.write(f"{dataset['name']},{k},{kk},{order},{scaling},{metric},{mean},{std}\n")
