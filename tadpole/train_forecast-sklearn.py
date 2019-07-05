import sys
sys.path.append('..')


from os.path import join

from tadpole.io_tp import load_tadpole_data, write_submission_table
from tadpole.validation import get_test_subjects
from tadpole.submission import create_submission_table
from tadpole.models.sklearn import create_prediction_batch


# Script requires that TADPOLE_D1_D2.csv is in the parent directory.
# Change if necessary.
dataLocationLB1LB2 = '../data/'  # current directory

# TODO dev
tadpoleLB1LB2_file = join(dataLocationLB1LB2, 'TADPOLE_LB1_LB2.csv')
output_file = '../data/TADPOLE_Submission_SummerSchool2018_MixedFeeling3.csv'

print('Loading data ...')
LB_table, LB_targets = load_tadpole_data(tadpoleLB1LB2_file)

# LB_table: num_patients * num_measurement_p_patient x num_features
# LB_targets: num_patients * num_measurement_p_patient x num_targets
print('Generating forecasts ...')

# * Create arrays to contain the 84 monthly forecasts for each LB2 subject
n_forecasts = 7 * 12  # forecast 7 years (84 months).
lb2_subjects = get_test_subjects(LB_table)

# submission = []
# # Each subject in LB2
# for rid in lb2_subjects:
#     subj_data = LB_table.query('RID == @rid')
#     subj_targets = LB_targets.query('RID == @rid')
#
#     # *** Construct example forecasts
#     subj_forecast = create_submission_table([rid], n_forecasts)
#     subj_forecast = create_prediction(subj_data, subj_targets, subj_forecast)
#
#     submission.append(subj_forecast)
data_grouped = LB_table.groupby("RID")
targets_grouped = LB_targets.groupby("RID")

rids = list(data_grouped.groups.keys())
batch_forecast = create_submission_table(rids, n_forecasts)
batch_forecast = create_prediction_batch(LB_table, LB_targets, batch_forecast)

## Now construct the forecast spreadsheet and output it.
print('Constructing the output spreadsheet {0} ...'.format(output_file))
write_submission_table(batch_forecast, output_file)
