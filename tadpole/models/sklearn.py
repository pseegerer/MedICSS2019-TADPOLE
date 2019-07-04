import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def predict_diagnosis(data_forecast, most_recent_data):
    if most_recent_CLIN_STAT == 'NL':
        CNp, MCIp, ADp = 0.3, 0.4, 0.3
    elif most_recent_CLIN_STAT == 'MCI':
        CNp, MCIp, ADp = 0.1, 0.5, 0.4
    elif most_recent_CLIN_STAT == 'Dementia':
        CNp, MCIp, ADp = 0.15, 0.15, 0.7
    else:
        CNp, MCIp, ADp = 0.33, 0.33, 0.34

    # Use the same clinical status probabilities for all months
    data_forecast.loc[:, 'CN relative probability'] = CNp
    data_forecast.loc[:, 'MCI relative probability'] = MCIp
    data_forecast.loc[:, 'AD relative probability'] = ADp


def predict_adas13(data_forecast, most_recent_data, features_list):
    # * ADAS13 forecast: = most recent score, default confidence interval
    most_recent_ADAS13 = most_recent_data['ADAS13'].dropna().tail(1).iloc[0]
    adas_mask = (most_recent_data['AGE_AT_EXAM'].dropna() > 0) & (
                most_recent_data['ADAS13'].dropna() > 0)  # not missing: Ventricles and ICV
    x = most_recent_data['AGE_AT_EXAM'].dropna()[adas_mask]
    y = most_recent_data['ADAS13'].dropna()[adas_mask]
    lm = np.polyfit(x, y, 1)
    adas_p = np.poly1d(lm)

    adas_prediction = np.maximum(most_recent_ADAS13, adas_p(most_recent_data['AGE_AT_EXAM'].dropna().iloc[-1] +
                                                            data_forecast['Forecast Month'] / 12))
    data_forecast.loc[:, 'ADAS13'] = adas_prediction
    data_forecast.loc[:, 'ADAS13 50% CI lower'] = adas_prediction - 1
    data_forecast.loc[:, 'ADAS13 50% CI upper'] = adas_prediction + 1

    # Subject has no history of ADAS13 measurement, so we'll take a
    # typical score of 12 with wide confidence interval +/-10.


def predict_gp(x, gp, mean_regressor):
    # Predict GP and add dataset mean
    prediction, std = gp.predict(x, return_std=True) + mean_regressor.predict(x)
    return prediction, std


def predict_ventricles(data_forecast, most_recent_data):
    # * Ventricles volume forecast: = most recent measurement, default confidence interval
    most_recent_Ventricles_ICV = most_recent_data['Ventricles_ICV'].dropna().tail(1).iloc[0]

    vent_mask = (most_recent_data['Ventricles_ICV'].dropna() > 0) & (
                most_recent_data['AGE_AT_EXAM'].dropna() > 0)  # not missing: Ventricles and ICV
    x = most_recent_data['AGE_AT_EXAM'].dropna()[vent_mask]
    y = most_recent_data['Ventricles_ICV'].dropna()[vent_mask]

    # Regress the joint mean
    # x = x.values.reshape((-1, 1))
    # mean_regressor = LinearRegression()
    # mean_regressor.fit(x, y)
    #
    # plt.figure()
    # plt.scatter(x, y, c="b")
    # x_min = x.min()
    # x_max = x.max()
    # plt.plot([x_min, x_max], [mean_regressor.predict(x_min.reshape((1, 1))), mean_regressor.predict(x_max.reshape((1, 1)))], "r",
    #          label="mean")
    # plt.legend()
    # plt.show()

    # Regress the individual subjects
    # Normalize the targets: how much does it deviate from the mean at this age?
    data_grouped = most_recent_data.dropna()[vent_mask].groupby("RID")

    # Go through all the subjects and build up huge feature matrix
    features = list()
    targets = list()
    groups = list()
    for ctr, (rid, subject) in enumerate(data_grouped):
        x = subject[["Entorhinal", "Fusiform", "MidTemp"]]
        num_measurements = len(x)
        if num_measurements < 2:
            print(f"Skipping {rid}")
            continue
        t = subject["AGE_AT_EXAM"].values
        vs = subject["Ventricles_ICV"].values
        y = subject['Ventricles_ICV']

        xx = x.values

        # Build up the feature vectors: x1 + (t' - t1) (+ is concat)
        # Target is v' - v1
        # Bigrams: x1 + x2 + (t2 - t1) + (t' - t1) + (v2 - v1)
        features_subject = list()
        targets_subject = list()
        for i in range(num_measurements):
            for j in range(i + 1, num_measurements):
                f = np.array((list(xx[i]) + [(t[j] - t[i])]))
                target = vs[j] - vs[i]
                features_subject.append(f)
                targets_subject.append(target)
                groups.append(rid)
        features_subject = np.stack(features_subject)
        features.append(features_subject)
        targets.extend(targets_subject)
    features = np.vstack(features)

    reg = RandomForestRegressor(n_jobs=3)
    param_grid = {"max_depth": [None, 4, 32], "n_estimators": [10, 100, 1000]}
    grid_search = GridSearchCV(reg, param_grid)
    grid_search.fit(features, y=targets, groups=groups)
    best_estimator = grid_search.best_estimator_
    print(pd.DataFrame(grid_search.cv_results_)[["mean_test_score", "mean_train_score"]])

    def predict_():
        # Get future time points
        dates_forecast = x[-1] + data_forecast[data_forecast["RID"] == rid]["Forecast Month"] / 12
        # TODO reshape should be generic
        vent_forecast, std = predict_gp(dates_forecast.values.reshape((-1, 1)), gp, mean_regressor)

        # TODO what index does it have?
        data_forecast.loc[data_forecast["RID"] == rid, 'Ventricles_ICV'] = vent_forecast

        # 50% CI. Phi(50%) = 0.75 -> 50% of the data lie within 0.75 * sigma around the mean
        data_forecast.loc[rid, 'Ventricles_ICV 50% CI lower'] = vent_forecast - 0.75 * std
        data_forecast.loc[rid, 'Ventricles_ICV 50% CI upper'] = vent_forecast + 0.75 * std

        # Plot for 1D regression
        # plt.figure()
        # xx = np.linspace(x_min, x_max, n_forecast, 100)
        # ygp = gp.predict(xx.reshape((-1, 1))) + mean_regressor.predict(xx.reshape((-1, 1))) # TODO for now it's the mean, can also return confidence
        # plt.plot(xx, ygp, "r")
        # plt.scatter(x, y, c="b")
        # plt.scatter(date_forecast, vent_forecast, c="g")
        # plt.title(f"Subject {rid}")
        # plt.show()


def create_prediction_batch(train_data, train_targets, data_forecast):
    """Create a linear regression prediction that does a first order
    extrapolation in time of ADAS13 and ventricles.

    :param train_data: Features in training data.
    :type train_data: pd.DataFrame
    :param train_targets: Target in trainign data.
    :param pd.DataFrame
    :param data_forecast: Empty data to insert predictions into
    :type data_forecast: pd.DataFrame
    :return: Data frame in same format as data_forecast.
    :rtype: pd.DataFrame
    """
    # * Clinical status forecast: predefined likelihoods per current status

    # List of features that are used for prediction
    features_list = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "WholeBrain"]
    most_recent_data = pd.concat((train_targets, train_data[['EXAMDATE', 'AGE_AT_EXAM'] + features_list]), axis=1).sort_values(by='EXAMDATE')
    most_recent_CLIN_STAT = most_recent_data['CLIN_STAT'].dropna().tail(1).iloc[0]



    # predict_diagnosis(data_forecast, most_recent_data)
    # predict_adas13(data_forecast, most_recent_data)
    predict_ventricles(data_forecast, most_recent_data)

    return data_forecast
