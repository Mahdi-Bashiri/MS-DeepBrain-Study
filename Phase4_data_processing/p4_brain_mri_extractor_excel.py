import os
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import json
from pathlib import Path
import re
from collections import defaultdict


class BrainMRIDataExtractor:
    def __init__(self, masks_directory, hc_csv_path=None, ms_csv_path=None):
        """
        Initialize the data extractor

        Args:
            masks_directory: Path to directory containing .npy and .pkl files
            hc_csv_path: Path to HC patient data CSV
            ms_csv_path: Path to MS patient data CSV
        """
        self.masks_directory_1 = Path(masks_directory[0])
        self.masks_directory_2 = Path(masks_directory[1])
        self.hc_csv_path = hc_csv_path
        self.ms_csv_path = ms_csv_path
        self.patient_info = {}
        self.results_all = []
        self.results_raw = []
        self.results_processed = []

    def load_patient_demographics(self):
        """Load patient demographics from CSV files"""
        print("Loading patient demographics...")

        if self.hc_csv_path and os.path.exists(self.hc_csv_path):
            hc_data = pd.read_csv(self.hc_csv_path)
            # Assuming columns are PatientID, PatientSex, PatientAge
            for _, row in hc_data.iterrows():
                patient_id = str(row.iloc[0])  # First column - PatientID
                sex = 0 if row.iloc[1] == 'F' else 1  # Second column - PatientSex (F=0, M=1)
                age = row.iloc[2]  # Third column - PatientAge
                self.patient_info[patient_id] = {
                    'age': age,
                    'sex': sex,
                    'group': 'HC'
                }

        if self.ms_csv_path and os.path.exists(self.ms_csv_path):
            ms_data = pd.read_csv(self.ms_csv_path)
            for _, row in ms_data.iterrows():
                patient_id = str(row.iloc[0])  # First column - PatientID
                sex = 0 if row.iloc[1] == 'F' else 1  # Second column - PatientSex (F=0, M=1)
                age = row.iloc[2]  # Third column - PatientAge
                self.patient_info[patient_id] = {
                    'age': age,
                    'sex': sex,
                    'group': 'MS'
                }

        print(f"Loaded demographics for {len(self.patient_info)} patients")

    def extract_patient_id_from_filename(self, filename):
        """Extract patient ID from filename"""
        # Filename format: 101228_..._8_processed_prediction_transformed_final.npy
        # First 6 digits are patient ID
        match = re.match(r'^(\d{6})', filename)
        if match:
            return match.group(1)
        return None

    def process_masks_and_results(self):
        """Process all mask files and extract data"""
        print("Processing mask files...")

        # Group files by patient and type (raw/processed)
        patient_files_all = defaultdict(list)
        patient_files_raw = defaultdict(list)
        patient_files_processed = defaultdict(list)

        for masks_directory_ in [self.masks_directory_1, self.masks_directory_2]:

            skull_masks_directory = masks_directory_ / 'FLAIR_Preprocessed' / 'Skull_Masks'
            brain_masks_directory = masks_directory_ / 'FLAIR_Preprocessed' / 'Brain_Masks'
            vent_masks_directory = masks_directory_ / 'FLAIR_Preprocessed' / 'Vent_Masks'

            prei_masks_directory = masks_directory_ / 'FLAIR_Preprocessed' / 'Sub_Masks' / 'peri_masks'
            para_masks_directory = masks_directory_ / 'FLAIR_Preprocessed' / 'Sub_Masks' / 'para_masks'
            juxt_masks_directory = masks_directory_ / 'FLAIR_Preprocessed' / 'Sub_Masks' / 'juxt_masks'

            pickle_directory = masks_directory_ / 'FLAIR_Preprocessed' / 'Sub_Masks' / 'pickles'
                
            # Find all .npy files
            nifti_files = list(brain_masks_directory.glob("*.gz"))
            print(f"Found {len(nifti_files)} .nii.gz files")

            for nifti_file in nifti_files:
                patient_id = self.extract_patient_id_from_filename(nifti_file.name)
                if patient_id:
                    # Find corresponding .pkl file
                    
                    skull_file = skull_masks_directory / (patient_id + '_skull_mask.nii.gz')
                    brain_file = brain_masks_directory / (patient_id + '_brain_mask.nii.gz')
                    vent_file = vent_masks_directory / (patient_id + '_vent_mask.nii.gz')

                    peri_file = prei_masks_directory / (patient_id + '_peri_mask.nii.gz')
                    para_file = para_masks_directory / (patient_id + '_para_mask.nii.gz')
                    juxt_file = juxt_masks_directory / (patient_id + '_juxt_mask.nii.gz')

                    pkl_file = pickle_directory / (patient_id + '_results.pkl')

                    if pkl_file.exists():
                        # Determine if this is raw or processed
                        patient_files_list = [skull_file, brain_file, vent_file, peri_file, para_file, juxt_file, pkl_file]
                        patient_files_all[patient_id].append(patient_files_list)

                        # if '_raw_' in nifti_file.name:
                        #     patient_files_raw[patient_id].append((nifti_file, pkl_file))
                        # elif '_processed_' in nifti_file.name:
                        #     patient_files_processed[patient_id].append((nifti_file, pkl_file))
                        # else:
                        #     print(f"Warning: Could not determine type (raw/processed) for {nifti_file.name}")
                    else:
                        print(f"Warning: No results file found for {patient_id}")

        print(f"Found {len(patient_files_all)} patients with raw data")
        # print(f"Found {len(patient_files_raw)} patients with raw data")
        # print(f"Found {len(patient_files_processed)} patients with processed data")

        # Process all data
        print("\nProcessing ALL mask data...")
        self.results_all = []
        for patient_id, files in patient_files_all.items():
            self.process_patient_data(patient_id, files, 'all')

        # # Process raw data
        # print("\nProcessing RAW mask data...")
        # self.results_raw = []
        # for patient_id, files in patient_files_raw.items():
        #     self.process_patient_data(patient_id, files, 'raw')

        # # Process processed data
        # print("\nProcessing PROCESSED mask data...")
        # self.results_processed = []
        # for patient_id, files in patient_files_processed.items():
        #     self.process_patient_data(patient_id, files, 'processed')

    def process_patient_data(self, patient_id, files, data_type):
        """Process data for a single patient"""
        # Initialize patient totals
        total_skull_area = 0
        total_brain_area = 0
        total_vent_area = 0
        total_peri_area = 0
        total_para_area = 0
        total_juxt_area = 0
        total_wmh_area = 0

        print(f"\t Processing patient {patient_id} with {len(files)} files...\n")

        for skull_file, brain_file, vent_file, peri_file, para_file, juxt_file, pkl_file in files:
            try:
                # Load the mask files
                skull_masks, skull_img = self.load_nifti(skull_file)
                brain_masks, brain_img = self.load_nifti(brain_file)
                vent_masks, vent_img = self.load_nifti(vent_file)
                peri_masks, peri_img = self.load_nifti(peri_file)
                para_masks, para_img = self.load_nifti(para_file)
                juxt_masks, juxt_img = self.load_nifti(juxt_file)

                # Load the results
                with open(pkl_file, 'rb') as f:
                    results_data = pickle.load(f)

                # Calculate areas from the masks
                # Color codes from your original code:
                # 'red': (255, 0, 0) - peri_found
                # 'orange': (255, 165, 0) - para_found
                # 'yellow': (255, 255, 0) - juxt_found
                # 'blue': (0, 0, 255) - vent_mask
                # 'torq': (0, 165, 255) - csf_mask

                # # Count pixels for each color/region
                # red_pixels = np.sum(np.all(rgb_mask == [255, 0, 0], axis=2))
                # orange_pixels = np.sum(np.all(rgb_mask == [255, 165, 0], axis=2))
                # yellow_pixels = np.sum(np.all(rgb_mask == [255, 255, 0], axis=2))
                # blue_pixels = np.sum(np.all(rgb_mask == [0, 0, 255], axis=2))

                # Add to totals
                total_skull_area += np.sum(skull_masks)
                total_brain_area += np.sum(brain_masks)
                total_vent_area += np.sum(vent_masks)
                total_peri_area += np.sum(peri_masks)
                total_para_area += np.sum(para_masks)
                total_juxt_area += np.sum(juxt_masks)
                total_wmh_area += (np.sum(peri_masks) + np.sum(para_masks) + np.sum(juxt_masks))

            except Exception as e:
                print(f"Error processing {patient_id}: {e}")
                continue

        # Get patient demographics
        patient_demo = self.patient_info.get(patient_id, {
            'age': None,
            'sex': None,
            'group': None
        })

        # Store results in appropriate list
        result_data = {
            'PatientID': patient_id,
            'PatientAge': patient_demo['age'],
            'PatientSex': patient_demo['sex'],
            'StudyGroup': patient_demo['group'],
            'TotalSkullArea': total_skull_area,
            'TotalIntracranialArea': total_brain_area,
            'TotalVentricleArea': total_vent_area,
            'TotalWMHArea': total_wmh_area,
            'TotalPeriArea': total_peri_area,
            'TotalParaArea': total_para_area,
            'TotalJuxtArea': total_juxt_area
        }

        if data_type == 'all':
            self.results_all.append(result_data)
        elif data_type == 'raw':
            self.results_raw.append(result_data)
        else:
            self.results_processed.append(result_data)

    def load_nifti(self, file_path):
        """Load a NIfTI file and return the image data and the nibabel object."""
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img

    def save_nifti(self, data, ref_img, out_path):
        """Save data as a NIfTI file using a reference image for header/affine."""
        new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
        nib.save(new_img, out_path)
        print(f"Saved pre-processed data to {out_path}")

    def save_results(self, results_list, output_path, format='csv', data_type=''):
        """Save results to file"""
        if not results_list:
            print(f"No {data_type} results to save!")
            return None

        df = pd.DataFrame(results_list)

        # Sort by PatientID
        df = df.sort_values('PatientID')

        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
            print(f"{data_type.title()} results saved to {output_path}")
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
            print(f"{data_type.title()} results saved to {output_path}")
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
            print(f"{data_type.title()} results saved to {output_path}")

        # Print summary statistics
        print(f"\n=== {data_type.upper()} DATA SUMMARY STATISTICS ===")
        print(f"Total patients processed: {len(df)}")
        if 'StudyGroup' in df.columns:
            group_counts = df['StudyGroup'].value_counts()
            print(f"Study groups: {group_counts.to_dict()}")

        print(f"\n{data_type.title()} area statistics (in pixels):")
        numeric_cols = ['TotalSkullArea', 'TotalIntracranialArea', 'TotalVentricleArea', 'TotalWMHArea', 'TotalPeriArea', 'TotalParaArea', 'TotalJuxtArea']
        print(df[numeric_cols].describe())

        return df


def main():
    """Main execution function"""


    # Configuration - UPDATE THESE PATHS

    masks_directory_hc = f"/mnt/d/TEMP_P4/HC_COHORT_PREP_prepared"
    masks_directory_ms = f"/mnt/d/TEMP_P4/MS_COHORT_PREP_prepared"
    hc_csv_path = "/mnt/d/TEMP_P4/Patient_Flair_data_HC.csv"
    ms_csv_path = "/mnt/d/TEMP_P4/Patient_Flair_data_MS.csv"

    output_path_all = f"/mnt/d/TEMP_P4/brain_mri_analysis_results_ALL.csv"
    output_path_raw = f"/mnt/d/TEMP_P4/brain_mri_analysis_results_RAW.csv"
    output_path_processed = f"/mnt/d/TEMP_P4/brain_mri_analysis_results_PROCESSED.csv"

    # Initialize extractor
    extractor = BrainMRIDataExtractor(
        masks_directory=[masks_directory_hc, masks_directory_ms],
        hc_csv_path=hc_csv_path,
        ms_csv_path=ms_csv_path
    )

    try:
        # Load patient demographics
        extractor.load_patient_demographics()

        # Process all masks and results
        extractor.process_masks_and_results()

        # Save ALL results
        if extractor.results_all:
            results_df_all = extractor.save_results(extractor.results_all, output_path_all, format='csv',
                                                    data_type='all')
            # Also save as Excel for easier viewing
            excel_path_all = output_path_all.replace('.csv', '.xlsx')
            extractor.save_results(extractor.results_all, excel_path_all, format='excel', data_type='all')

        # Save RAW results
        if extractor.results_raw:
            results_df_raw = extractor.save_results(extractor.results_raw, output_path_raw, format='csv',
                                                    data_type='raw')
            # Also save as Excel for easier viewing
            excel_path_raw = output_path_raw.replace('.csv', '.xlsx')
            extractor.save_results(extractor.results_raw, excel_path_raw, format='excel', data_type='raw')

        # Save PROCESSED results
        if extractor.results_processed:
            results_df_processed = extractor.save_results(extractor.results_processed, output_path_processed,
                                                        format='csv', data_type='processed')
            # Also save as Excel for easier viewing
            excel_path_processed = output_path_processed.replace('.csv', '.xlsx')
            extractor.save_results(extractor.results_processed, excel_path_processed, format='excel',
                                data_type='processed')

        print(f"\n=== PROCESSING COMPLETE ===")
        if extractor.results_all:
            print(f"ALL CSV file saved: {output_path_all}")
            print(f"ALL Excel file saved: {excel_path_all}")
        if extractor.results_raw:
            print(f"RAW CSV file saved: {output_path_raw}")
            print(f"RAW Excel file saved: {excel_path_raw}")
        if extractor.results_processed:
            print(f"PROCESSED CSV file saved: {output_path_processed}")
            print(f"PROCESSED Excel file saved: {excel_path_processed}")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()