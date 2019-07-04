import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm

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


def get_mixed_effects(subject, feature_list):
    baseline_features = subject.iloc[0][feature_list]
    t_bl = subject["AGE_AT_EXAM"].min()
    t = subject["AGE_AT_EXAM"] - t_bl
    y = subject['Ventricles_ICV']
    fixed_effect = pd.DataFrame(t)
    fixed_effect["t_bl"] = t_bl
    fixed_effect = pd.concat([fixed_effect, pd.DataFrame([baseline_features] * len(fixed_effect), index=fixed_effect.index)], axis=1,
                             join="inner")
    random_effect = t

    # Add intercept
    fixed_effect["intercept"] = 1.0
    random_effect["intercept"] = 1.0

    return fixed_effect, random_effect, y


def predict_lme(result, rid, year, test_subject):
    fe_params = result.fe_params.values
    re_params = result.random_effects[rid].values
    test_subject["AGE_AT_EXAM"] = year
    prediction = test_subject @ fe_params + np.array([year]) @ re_params
    return prediction


def predict_ventricles(data_forecast, most_recent_data, feature_list):
    # * Ventricles volume forecast: = most recent measurement, default confidence interval
    most_recent_Ventricles_ICV = most_recent_data['Ventricles_ICV'].dropna().tail(1).iloc[0]

    vent_mask = (most_recent_data['Ventricles_ICV'].dropna() > 0) & (
                most_recent_data['AGE_AT_EXAM'].dropna() > 0)  # not missing: Ventricles and ICV
    test_subject = most_recent_data['AGE_AT_EXAM'].dropna()[vent_mask]
    y = most_recent_data['Ventricles_ICV'].dropna()[vent_mask]

    # Regress the individual subjects
    # Normalize the targets: how much does it deviate from the mean at this age?
    data_grouped = most_recent_data.dropna()[vent_mask].groupby("RID")

    # Go through all the subjects and build up huge feature matrix
    fixed_effects = list()
    random_effects = list()
    groups = list()
    ys = list()
    for ctr, (rid, subject) in enumerate(data_grouped):
        num_measurements = len(subject)

        fixed_effect, random_effect, y = get_mixed_effects(subject, feature_list)

        fixed_effects.append(fixed_effect)
        random_effects.append(random_effect)
        groups.extend(num_measurements * [rid])
        ys.append(y)

    fixed_effects = pd.concat(fixed_effects, axis=0)
    random_effects = pd.concat(random_effects, axis=0)

    ys = pd.concat(ys, axis=0)

    model = MixedLM(ys, fixed_effects, groups, exog_re=random_effects)
    result = model.fit()
    print(result.summary())

    for rid, test_subject in tqdm.tqdm(data_grouped):
        t_bl = test_subject["AGE_AT_EXAM"].min()
        fixed_effect, _, _ = get_mixed_effects(test_subject, feature_list)
        # Get future time points
        dates_forecast = t_bl + data_forecast[data_forecast["RID"] == rid]["Forecast Month"] / 12

        # TODO reshape should be generic
        vent_forecasts = list()
        vent_std = list()
        std = 0.01
        for date_forecast in dates_forecast:
            vent_forecasts.append(predict_lme(result, rid, date_forecast, fixed_effect.iloc[0]))
            vent_std.append(std)  # TODO

        # TODO what index does it have?
        data_forecast.loc[data_forecast["RID"] == rid, 'Ventricles_ICV'] = vent_forecasts

        # 50% CI. Phi(50%) = 0.75 -> 50% of the data lie within 0.75 * sigma around the mean
        data_forecast.loc[data_forecast["RID"] == rid, 'Ventricles_ICV 50% CI lower'] = np.array(vent_forecasts) - 0.75 * std
        data_forecast.loc[data_forecast["RID"] == rid, 'Ventricles_ICV 50% CI upper'] = np.array(vent_forecasts) + 0.75 * std
    return data_forecast



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
    data_forecast = predict_ventricles(data_forecast, most_recent_data, features_list)

    return data_forecast
