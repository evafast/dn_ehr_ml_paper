"""
python3 preprocessing.py --before_biopsy_flag True --years_before_biopsy 1
python3 preprocessing.py --before_biopsy_flag True --years_before_biopsy 0
python3 preprocessing.py --before_biopsy_flag True --years_before_biopsy 3
python3 preprocessing.py --before_biopsy_flag True --years_before_biopsy 5 --gap_years 2
python3 preprocessing.py --before_biopsy_flag True --years_before_biopsy 7 --gap_years 1
python3 preprocessing.py --before_biopsy_flag True --years_before_biopsy 3 --gap_years 1 --required_labs "['eGFR (GFB)', 'UPCR (GFB)', 'UACR (GFB)']"

"""
import pandas as pd
from gfbutils.functions import log, skydb
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import ast

# Initiate common variables
logger = log.init_logger("preporcessing-logs")

def parse_args():
    """
    Use the argparse library to provide special help text and simplify argument handling

    :return: tuple
    """
    parser = argparse.ArgumentParser(
        description='Run Tuttle Pre-processing Code. This can take multiple arguments.')
    parser.add_argument('--s3_bucket_name', dest='s3_bucket_name', required=False,
                        default='kga-warehouse-prod-us-east-2',
                        help="The AWS S3 bucket name where the raw input files are available.")
    parser.add_argument('--before_biopsy_flag', dest='before_biopsy_flag', required=False,
                        default=False, type=bool,
                        help="The flag to aggregate results only before biopsy date. If False, All records are considered.")
    parser.add_argument('--years_before_biopsy', dest='years_before_biopsy', required=False,
                        default=0, type=int,
                        help="The Number of years of data to be considered before the biopsy was taken.")
    parser.add_argument('--gap_years', dest='gap_years', required=False,
                        default=0, type=int,
                        help="The Number of years of data to be ignored before the date of biopsy.")
    parser.add_argument('--required_labs', dest='required_labs', required=False,
                        default=['25-Hydroxy D3', 'TSH', 'UACR (GFB)', 'Platelets', 'PTT', 'Hemoglobin A1C', 'ALT', 'Fasting Glucose', 'TSAT', 'PT', 'Indirect Bilirubin', 'Glucose in Urinalysis', 'Hematocrit', 'B-2-MICROGLOBULIN', 'Cholesterol/HDL ratio', 'BUN', 'Alkaline Phosphatase', 'THROMBIN TIME', 'Creatinine Clearance', 'AST', '25-Hydroxy D, Total', 'FREE T3', 'C4', 'Microalbumin', 'CK-MB', 'MPV', 'iPTH', 'Chloride', 'ANION GAP', 'Hep B Surface Antibody', 'Bilirubin in Urinalysis', 'APTT', 'Non-HDL Cholesterol', 'C3', 'Serum Albumin % of Serum Protein ', 'Eosinophils (absolute)', 'Eosinophils (percent)', 'Calcium, Ionized', 'BUN/CREATININE RATIO', 'INR', 'RDW', 'TROPONIN T', 'MCHC', 'IgG', 'Sodium', 'Total Bilirubin', 'PREALBUMIN', 'Cyclosporine Level, HPLC, External', 'Cyclosporine Level, TDX, External', 'Cyclosporine Lvl', 'CYCLOSPORINE - EXT', 'CO2', 'Lymphocytes (absolute)', 'Lymphocytes (percent)', 'Triglycerides', 'Serum Iron', 'NRBC', 'IgA', 'ALKALINE PHOSPHATASE ISOENZYMES', 'ALKALINE PHOSPHATASE (OSL)', 'ALKALINE PHOSPHATASE TOTAL', 'T4', 'Albumin/Globulin Ratio', 'D-Dimer', 'Uric Acid', 'Transferrin', 'C-Peptide',
                                    'Ferritin', 'T3', 'UPCR (GFB)', 'Total Protein', 'Serum Albumin', 'RBC', '25-Hydroxy D2', 'Fasting Insulin', 'Basophils (percent)', 'Basophils (absolute)', 'Total Cholesterol', 'HEP B CORE AB', 'Calcium', 'Calcium Corrected', 'Monocytes (absolute)', 'Monocytes (percent)', 'C-Reactive Protein', 'IgM', 'Calcium', 'MCH', 'Serum Creatinine', 'UACR', 'UPCR', 'TRIGLYCERIDE', 'O2 Saturation', 'BNP', 'TROPONIN I', 'WBC', 'Urine Creatinine', 'Vitamin B12', 'iPTH', 'PTH RELATED PROTEIN', 'PTH, intact', 'PTH INTACT', 'PTH Intact, External', 'PTH Related Protein', 'PTH Baseline', 'PTH, Intraoperative', 'PTH, INTACT', 'PTH-RP', 'CALCIUM FOR PTH', 'PTH,INTRAOPERATIVE,POC', 'PTH, INTACT INTRAOPERATIVE', 'PTH (INTACT ASSAY), SERUM(LBC)', 'PTH, INTACT(LBC)', 'PTH-RELATED PROTEIN (PTH-RP)', 'Serum Cortisol', 'CEA', 'TRIGLYCERIDE/HDL', 'Serum Albumin', 'RDW-SD', 'Potassium', 'Neutrophils (absolute)', 'Cyclosporine', 'LDL', 'Immature Granulocytes (absolute)', 'Immature Granulocytes (percent)', 'TIBC', 'HDL', 'Hemoglobin', 'Magnesium', 'Random Glucose', 'VLDL', 'MCV', 'eGFR', 'Phosphorus', 'Direct Bilirubin', 'eGFR (GFB)'], 
                        help="The Labs to be considered.")
    parser.add_argument('--required_procedures', dest='required_procedures', required=False,
                        default="All", help="The procedures to be considered.")
    args = parser.parse_args()

    return args.s3_bucket_name, args.before_biopsy_flag, args.years_before_biopsy, args.gap_years, args.required_labs, args.required_procedures


def get_num_of_days(start_date, end_date):
    """This function will be used to determine the days a patient was in hospital for the visit as per start and end date of visit.

    Args:
        start_date (String): The start date of the visit
        end_date (String): The end date of the visit

    Returns:
        Float: The days the patient was in hospital during the visits
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    hospital_visits_duration = "{:.4f}".format(
        ((end_date - start_date).total_seconds()) / 86400)
    return float(hospital_visits_duration)


def get_min_max_df(input_df, required_col_list):
    """This function will be used to add 2 columns, 1 for max value and 1 for min values

    Args:
        input_df (DF): The input DF with the columns to be worked upon
        required_col_list (List): The list of columns on which the min / max should be applied.

    Returns:
        DF: The dataframe with all the min/max columns and the columns not included as part of input list but in input DF.
    """
    input_df = input_df.fillna("")
    final_max_min_list = []
    for name_of_the_group, group in input_df.groupby(by=['subject_id', 'gfb_subject_id']):
        subject_id = name_of_the_group[0]
        gfb_subject_id = name_of_the_group[1]
        temp_dict = {
            'subject_id': subject_id,
            'gfb_subject_id': gfb_subject_id
        }
        for col_name in required_col_list:
            max_col_name_var = "max_" + col_name.replace(" ", "")
            min_col_name_var = "min_" + col_name.replace(" ", "")
            for spcl_char in ["-", "(", ')', '/', ',', '%']:
                max_col_name_var = max_col_name_var.replace(spcl_char, "_")
                min_col_name_var = min_col_name_var.replace(spcl_char, "_")

            max_col_name_var = max_col_name_var.rstrip("_")
            min_col_name_var = min_col_name_var.rstrip("_")

            temp_list_of_vals = list(set(filter(None, group[col_name].to_list())))
            temp_list_of_vals = [x for x in temp_list_of_vals if ("<" not in str(x)) and (">" not in str(x))]
            # Convert strings to floats
            temp_list_of_vals = list(map(float, temp_list_of_vals))

            vars()[max_col_name_var] = None
            vars()[min_col_name_var] = None
            if len(temp_list_of_vals) > 0:
                vars()[max_col_name_var] = max(temp_list_of_vals)
                vars()[min_col_name_var] = min(temp_list_of_vals)

            temp_dict[str(max_col_name_var)] = eval(max_col_name_var)
            temp_dict[str(min_col_name_var)] = eval(min_col_name_var)

        final_max_min_list.append(temp_dict)

    return pd.DataFrame(final_max_min_list)


def get_median_df(input_df, group_by_col_list, drop_col_list):
    """This function will be used for aggregating the dataframe on median

    Args:
        input_df (DF): The input dataframe to be median
        group_by_col_list (List]): The cols on which we need to group by
        drop_col_list (List): The cols we want to remove from DF

    Returns:
        Pandas DF: The dataframe after aggregation (median)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        temp_agg_df = input_df.groupby(by=group_by_col_list).median()

    temp_agg_df = temp_agg_df.reset_index()

    if drop_col_list:
        temp_agg_df.drop(columns=drop_col_list, inplace=True)
    return temp_agg_df


def filter_records_before_biopsy(input_df, date_col_name, biopsy_report_df, years_before_biopsy, gap_years):
    """This function will be used to filter records as per biopsy date and years before biopsy

    Args:
        input_df (DF): The dataframe that needs to be filterred
        date_col_name (String): The date col which should be used to filter before biopsy date
        biopsy_report_df (DF): The biopsy report dataframe
        years_before_biopsy (Int): The years of data we want before the biopsy date. 0 means all.
        gap_years (Int): The years of data we want to ignore before the biopsy date. 0 means all. 1 means the data should be collected till 1 year before biopsy date.
                        Ex: Patient History : 2010-2020
                        Biopsy Date : 1 Jan 2019
                        Gap Years : 2
                        Data to be collected : 2010 - 1 Jan 2017
        
    Returns:
        Pandas DF: The filtered day with records only before biopsy date for the patient.
    """
    # Convert the date col to date object for comparison
    input_df[date_col_name] = pd.to_datetime(input_df[date_col_name]).dt.date
    biopsy_report_df['biopsy_date'] = pd.to_datetime(biopsy_report_df['biopsy_date']).dt.date
    df_before_biopsy = pd.DataFrame()
    for record in biopsy_report_df.to_dict(orient="records"):
        if years_before_biopsy > 0:
            records_gt_than_date = record['biopsy_date'].replace(year=record['biopsy_date'].year-years_before_biopsy)
            temp_before_biopsy_df = input_df[(input_df['subject_id'] == record['subject_id']) & (input_df[date_col_name] < record['biopsy_date']) & (input_df[date_col_name] > records_gt_than_date)]
        else:
            temp_before_biopsy_df = input_df[(input_df['subject_id'] == record['subject_id']) & (input_df[date_col_name] < record['biopsy_date'])]

        if gap_years > 0:
            records_lt_than_date = record['biopsy_date'].replace(year=record['biopsy_date'].year-gap_years)
            temp_before_biopsy_df = temp_before_biopsy_df[temp_before_biopsy_df[date_col_name] < records_lt_than_date]

        if len(temp_before_biopsy_df) > 0:
            df_before_biopsy = pd.concat([df_before_biopsy, temp_before_biopsy_df], ignore_index=True)

    return df_before_biopsy

def get_procedures_df():
    """This function will be used to generate the transformed procedures for each subject with True Flags if available.

    Returns:
        [type]: [description]
    """
    procedures_df = pd.DataFrame()
    procedures_df[['subject_id',  'gfb_subject_id', 'procedure_code', 'procedure_code_type', 'procedure_code_source',
                    'procedure_code_date']] = icd_codes_df[['subject_id',  'gfb_subject_id', 'code', 'code_type', 'code_source', 'code_date']]

    if required_procedures != "All":
        procedures_df = procedures_df[procedures_df['procedure_code'].isin(required_procedures)]
        logger.info("Filtered procedures as per codes passed in command line.")

    # Filtered Proc DF with only codes
    transformed_list = []
    for subject_and_gfb_subject, group in procedures_df[['subject_id',  'gfb_subject_id', 'procedure_code']].groupby(by=['subject_id',  'gfb_subject_id']):
        transposed_dict = {
            'subject_id': subject_and_gfb_subject[0],
            'gfb_subject_id': subject_and_gfb_subject[1]}
        available_procs = list(set(group['procedure_code'].to_list()))
        for key in available_procs:
            transposed_dict[key] = True

        transformed_list.append(transposed_dict)

    return pd.DataFrame(transformed_list)


def get_medications_df(medications_df):
    """This will aggregate the medications.

    Args:
        medications_df (DF): The medications dataframe

    Returns:
        Pandas DF: The aggregated DF
    """
    medications_df = medications_df[~medications_df['gfb_medication_class'].isna()]
    medications_df = medications_df[['gfb_subject_id', 'gfb_medication_class']]
    medications_df = pd.crosstab(medications_df['gfb_subject_id'], medications_df['gfb_medication_class'])
    medications_df['medication_count'] = medications_df.sum(axis=1)
    medications_df = pd.concat([medications_df.iloc[:,:-1] > 0, medications_df['medication_count']], axis=1)

    return pd.DataFrame(medications_df)



def get_vitals_df():
    """
    This function will be used to create a vitals df from visit_observations_df along with min/max values and hospital visit details (needs fix)
    """
    vitals_df = visit_observations_df[['subject_id', 'gfb_subject_id', 'height_cm', 'weight_lb', 'bp_sit_sys', 'bp_sit_dias', 'bmi', 'bmi_percent', 'pulse', 'hypertension_status']]
    vitals_median_df = get_median_df(vitals_df, ['subject_id', 'gfb_subject_id'], None)
    vitals_max_min_df = get_min_max_df(vitals_df, ['height_cm', 'weight_lb', 'bp_sit_sys', 'bp_sit_dias', 'bmi', 'bmi_percent', 'pulse', 'hypertension_status'])

    
    # Get hospital visit related columns details
    hospital_visit_observations_df = visit_observations_df[visit_observations_df['visit_type'].isin(['ER', 'Hospital Visit'])]
    # Add Visit Count for aggregation
    hospital_visit_observations_df['hospital_visits_count'] = 1
    # Add Days column
    hospital_visit_observations_df['hospital_visits_duration_days'] = hospital_visit_observations_df.apply(lambda x: get_num_of_days(str(x.visit_start_date), str(x.visit_end_date)), axis=1)
    hospital_visit_observations_df['hospital_visits_duration_days'] = "Fix Needed"
    # Group By
    hospital_visit_observations_df = hospital_visit_observations_df[['subject_id', 'gfb_subject_id', 'hospital_visits_count', 'hospital_visits_duration_days']]
    hospital_visit_observations_df = hospital_visit_observations_df.groupby(['subject_id', 'gfb_subject_id'])[['hospital_visits_count', 'hospital_visits_duration_days']].sum().reset_index()
    # Adding "Frequency_of_visits"
    # hospital_visit_observations_df['freq_of_visits'] = hospital_visit_observations_df['hospital_visits_count'] / visit_observations_df['hospital_visits_duration_days']
    hospital_visit_observations_df['freq_of_visits'] = "Fix Needed"

    # Merging temp dfs
    vitals_and_visits_df = pd.merge(vitals_median_df, vitals_max_min_df, on=['subject_id', 'gfb_subject_id'], how="left")
    vitals_and_visits_df = pd.merge(vitals_and_visits_df, hospital_visit_observations_df, on=['subject_id', 'gfb_subject_id'], how="left")

    return vitals_and_visits_df


def get_labs_df(input_required_labs):
    """This function will be used to transpose the labs and also aggregate and take min/max of the lab values.

    Returns:
        DF: Final labs df to be merged.
    """
    input_required_labs = ast.literal_eval(str(input_required_labs))
    req_labs_df = lab_results_df[lab_results_df['test_name'].isin(input_required_labs)]

    transformed_list = []
    for name_of_the_group, group in req_labs_df.groupby(by=['subject_id', 'gfb_subject_id', 'result_date']):
        subject_id, gfb_subject_id, result_date = [name_of_the_group[i] for i in (0, 1, 2)]
        transposed_dict = {
            'subject_id': subject_id,
            'gfb_subject_id': gfb_subject_id,
            'result_date': result_date
        }

        available_labs = list(set(group['test_name'].to_list()))
        for key in input_required_labs:
            if key in available_labs:
                transposed_dict[key] = group[group['test_name'].isin([key])]['result_value'].iloc[0]
            else:
                transposed_dict[key] = None
        transformed_list.append(transposed_dict)

    transposed_df = pd.DataFrame(transformed_list)

    # Get the min/max values for labs
    labs_max_min_df = get_min_max_df(transposed_df, input_required_labs)
    labs_median_df = get_median_df(transposed_df, ['subject_id', 'gfb_subject_id'], None)

    # Merge the temp DFs
    final_labs_df = pd.merge(labs_median_df, labs_max_min_df, on=['subject_id', 'gfb_subject_id'], how="left")

    return final_labs_df


if __name__ == "__main__":
    try:
        logger.info("Starting pre-processing step.")

        # Reading the arguments and setting up variables
        s3_bucket_name, before_biopsy_flag, years_before_biopsy, gap_years, required_labs, required_procedures = parse_args()
        logger.info("The S3 Bucket : %s ." % s3_bucket_name)
        logger.info("Do we consider results before Biopsy Date : %s ." % before_biopsy_flag)
        logger.info("The years of data to be considered before Biopsy Date : %s years." % years_before_biopsy)
        logger.info("The years of data to be ignored before Biopsy Date as Gap Years : %s years." % gap_years)

        # The tables from skydb actually needed for preprocessing
        skydb_table_mapping = {
            'biopsy_reports': None,
            'demographics': None,
            'disease_cohort': None,
            'visit_observations': 'visit_start_date',
            'lab_results': 'result_date',
            'icd_codes': 'code_date',
            'medications': 'start_date',
        }

        # This is a temp path used for downloading data as it becomes huge while pulling from skydb and it takes time.
        skydb_output_path = "/Users/spatel/Documents/GitLab/clinical-data-pipeline/batch_process/pre_processing/updated_tuttle/preprocessing/skydb_outputs"

        # Loop through each item in skydb_mapping and create a DF
        for table in skydb_table_mapping.keys():
            logger.info("Getting the data from skydb table : %s" % table)
            temp_df = skydb.get_clinical_data_df(gfb_subject_ids=[], category=table, study_names=["Tuttle"], aws_region="us-east-2", aws_profile="kgaprod")
            logger.info("The no. of records to gathered from skydb are: % s" % len(temp_df))
            if table != "biopsy_reports" and before_biopsy_flag and years_before_biopsy is not None and skydb_table_mapping[table] is not None:
                temp_df = filter_records_before_biopsy(temp_df, skydb_table_mapping[table], biopsy_reports_df, years_before_biopsy, gap_years)

            logger.info("The no. of records to be considered for this table are : %s" % len(temp_df))
            vars()[table + "_df"] = temp_df


        # This step is ideally not needed, but added to avoid warnings (df is not defined) in IDE for later code.
        biopsy_reports_df = biopsy_reports_df
        demographics_df = demographics_df
        disease_cohort_df = disease_cohort_df
        visit_observations_df = visit_observations_df
        lab_results_df = lab_results_df
        icd_codes_df = icd_codes_df
        medications_df = medications_df

        # Rename columns from demographics
        renaming_dem_cols = {'race': 'patient_race',
                            'gender': 'patient_gender',
                            'ethnicity': 'patient_ethnicity',
                            'place_of_origin': 'state'}

        demographics_df.rename(columns=renaming_dem_cols, inplace=True)
        demographics_df['ruca_code'] = demographics_df['ruca_code'].astype(str)
        
        logger.info("The column names are updated for demographics DF")
        
        # Get Biopsy Indication Column from the report column
        biopsy_reports_df['biopsy_indication'] = biopsy_reports_df['biopsy_report'].apply(lambda x: str(str(x).split("|")[0]).replace("Biopsy_Indication: ", "") if "Biopsy_Indication" in x else "")
        logger.info("The biopsy_indication column was parsed and added.")
        
        # Get "diabetic_retinopathy" status
        list_of_subjects_with_diab_ret = list(disease_cohort_df[disease_cohort_df['taxon_unique_name'].isin(['152. Diabetic Retinopathy'])]['subject_id'].unique())
        logger.info("List of subjects with Diabetic Retinopathy determined.")
        
        # Get Vitals Information
        final_vitals_visits_df = get_vitals_df()
        logger.info("The vitals df is created.")

        # Parse labs
        final_labs_df = get_labs_df(required_labs)
        logger.info("The labs df is created.")

        # Get Procedure info
        transposed_procedures_df = get_procedures_df()
        logger.info("The procedures DF is created.")

        # Get Medication info
        final_medications = get_medications_df(medications_df)
        logger.info("The medication DF is created.")

        
        # Merge and Get final Biopsy_reports file 
        logger.info("Starting to merge all the dataframes into a single DF.")
        final_df = pd.merge(demographics_df[['subject_id', 'gfb_subject_id',  'patient_race', 'patient_gender', 'patient_ethnicity', 'dob', 'state', 'ruca_code']], 
                            biopsy_reports_df[['subject_id', 'gfb_subject_id', 'biopsy_date', 'biopsy_indication']], 
                            on=['subject_id', 'gfb_subject_id'], how="left")

        final_df['diabetic_retinopathy'] = final_df['subject_id'].apply(lambda x: "Yes" if x in list_of_subjects_with_diab_ret else "No")
        final_df = pd.merge(final_df, final_vitals_visits_df, on=['subject_id', 'gfb_subject_id'], how="left")
        final_df = pd.merge(final_df, final_labs_df, on=['subject_id', 'gfb_subject_id'], how="left")
        final_df = pd.merge(final_df, transposed_procedures_df, on=['subject_id', 'gfb_subject_id'], how="left")
        final_df = pd.merge(final_df, final_medications, on=['gfb_subject_id'], how="left")

        logger.info("The final dataframe is created with %s records and %s columns." % (len(final_df), len(final_df.columns)))

        #Write Output to S3
        if before_biopsy_flag:
            output_parquet_file = "s3://%s/clinical_data_tuttle/preprocessed_data/ef_agg_before_biopsy_%s_years_%s_gap.parquet" % (s3_bucket_name, years_before_biopsy, gap_years)
            # output_parquet_file = "/Users/spatel/GitLab/clinical-data-pipeline/batch_process/pre_processing/tuttle/agg_before_biopsy_%s_years_%s_gap.parquet" % (years_before_biopsy, gap_years)
        else:
            output_parquet_file = "s3://%s/clinical_data_tuttle/preprocessed_data/ef_agg_data_all_data.parquet" % s3_bucket_name
            # output_parquet_file = "/Users/spatel/GitLab/clinical-data-pipeline/batch_process/pre_processing/tuttle/agg_data_all_data.parquet"

        final_df.to_parquet(output_parquet_file, index=False)
        logger.info("The final combined DF is written to the file : %s" % output_parquet_file)

        logger.info("The pre-processing is complete.")
        
    except:
        logger.error(
            "The pre-processing has failed. Kindly check.", exc_info=True)
