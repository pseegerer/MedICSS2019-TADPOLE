import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from statsmodels.regression.mixed_linear_model import MixedLM

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


def predict_ventricles(data_forecast, most_recent_data, feature_list):
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
    fixed_effects = list()
    random_effects = list()
    groups = list()
    ys = list()
    for ctr, (rid, subject) in enumerate(data_grouped):
        x = subject.iloc[0][feature_list]  # TODO take the baseline features
        num_measurements = len(subject)
        # TODO
        if num_measurements < 2:
            print(f"Skipping {rid}")
            continue

        t_bl = subject["AGE_AT_EXAM"].min()
        t = subject["AGE_AT_EXAM"] - t_bl
        y = subject['Ventricles_ICV']
        fixed_effect = pd.DataFrame(t)
        fixed_effect["t_bl"] = t_bl
        fixed_effect = pd.concat([fixed_effect, pd.DataFrame([x] * len(fixed_effect), index=fixed_effect.index)], axis=1, join="inner")

        fixed_effects.append(fixed_effect)
        random_effects.append(t)
        groups.extend(num_measurements * [rid])
        ys.append(y)

    fixed_effects = pd.concat(fixed_effects, axis=0)
    random_effects = pd.concat(random_effects, axis=0)
    fixed_effects["intercept"] = 1
    random_effects["intercept"] = 1

    ys = pd.concat(ys, axis=0)

    model = MixedLM(ys, fixed_effects, groups, exog_re=random_effects)
    result = model.fit()
    print(result.summary())

    def predict_lme(result, rid, year, test_subject):
        fe_params = result.fe_params.values
        re_params = result.random_effects[rid].values
        # test_subject = pd.concat(len(year) * [pd.DataFrame(test_subject).T], axis=0)
        test_subject["AGE_AT_EXAM"] = year
        prediction = test_subject @ fe_params + np.array([year]) @ re_params
        return prediction

    # test_subject = fixed_effects.iloc[0]
    # predict_lme(result, 3, 3, test_subject)

    for x in subjects_bl:
        t_bl = None
        test_subject = None
        # Get future time points
        dates_forecast = t_bl + data_forecast[data_forecast["RID"] == rid]["Forecast Month"] / 12
        # TODO reshape should be generic
        vent_forecast, std = predict_lme(result, rid, dates_forecast, test_subject)

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
    features_list = ["Hippocampus_bl", "Entorhinal_bl", "Fusiform_bl", "MidTemp_bl", "WholeBrain_bl"]
    most_recent_data = pd.concat((train_targets, train_data[['EXAMDATE', 'AGE_AT_EXAM'] + features_list]), axis=1).sort_values(by='EXAMDATE')
    most_recent_CLIN_STAT = most_recent_data['CLIN_STAT'].dropna().tail(1).iloc[0]



    # predict_diagnosis(data_forecast, most_recent_data)
    # predict_adas13(data_forecast, most_recent_data)
    predict_ventricles(data_forecast, most_recent_data, features_list)

    return data_forecast
