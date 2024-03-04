

# kNN
with open("./results/knn.csv", "w") as f:
    # first line in csv file
    f.write("dataset,k,weights,scaling,metric,mean,std\n")
    # iterate over datasets
    for dataset in links:
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
            ks = [2,5,7,10,20,30,40,50]
        # medium
        elif X.shape[0] < 50000:
            ks = [2,5,7,10,25,50,100,250]
        # large
        else:
            ks = [2,3,5,7,10,12,15,20,25]

        for k in ks:
            for weights in ["uniform","distance"]:
                for scaling in [0, 1]:
                    for metric in ["r2"]:
                        model = KNeighborsRegressor(n_neighbors=k, weights=weights)
                        if scaling:
                            reg = make_pipeline(standard_scaling, dn_scaling, model)
                            scores = cross_val_score(reg, X_train, y_train, scoring=metric, cv=10, n_jobs=4)  
                        else:
                            reg = make_pipeline(standard_scaling, model)
                            scores = cross_val_score(reg, X_train, y_train, scoring=metric, cv=10, n_jobs=4)
                        mean, std = scores.mean(), scores.std()
                        print(f"{dataset['name']},{k},{weights},{scaling},{metric},{mean},{std}")
                        f.write(f"{dataset['name']},{k},{weights},{scaling},{metric},{mean},{std}\n")
