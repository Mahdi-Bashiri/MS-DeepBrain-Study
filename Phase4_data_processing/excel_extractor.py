import os
import numpy as np
import pandas as pd
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
        self.masks_directory = Path(masks_directory)
        self.hc_csv_path = hc_csv_path
        self.ms_csv_path = ms_csv_path
        self.patient_info = {}
        self.results = []
        
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
        
        # Group files by patient
        patient_files = defaultdict(list)
        
        # Find all .npy files
        npy_files = list(self.masks_directory.glob("*.npy"))
        print(f"Found {len(npy_files)} .npy files")
        
        for npy_file in npy_files:
            patient_id = self.extract_patient_id_from_filename(npy_file.name)
            if patient_id:
                # Find corresponding .pkl file
                pkl_file = npy_file.with_suffix('').with_suffix('') + '_results.pkl'
                if pkl_file.exists():
                    patient_files[patient_id].append((npy_file, pkl_file))
                else:
                    print(f"Warning: No results file found for {npy_file.name}")
        
        print(f"Processing data for {len(patient_files)} patients")
        
        # Process each patient's data
        for patient_id, files in patient_files.items():
            self.process_patient_data(patient_id, files)
    
    def process_patient_data(self, patient_id, files):
        """Process data for a single patient"""
        # Initialize patient totals
        total_vent_area = 0
        total_wmh_area = 0
        total_peri_area = 0
        total_para_area = 0
        total_juxt_area = 0
        
        print(f"Processing patient {patient_id} with {len(files)} slices...")
        
        for npy_file, pkl_file in files:
            try:
                # Load the RGB mask
                rgb_mask = np.load(npy_file)
                
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
                
                # Count pixels for each color/region
                red_pixels = np.sum(np.all(rgb_mask == [255, 0, 0], axis=2))
                orange_pixels = np.sum(np.all(rgb_mask == [255, 165, 0], axis=2))
                yellow_pixels = np.sum(np.all(rgb_mask == [255, 255, 0], axis=2))
                blue_pixels = np.sum(np.all(rgb_mask == [0, 0, 255], axis=2))
                
                # Add to totals
                total_peri_area += red_pixels
                total_para_area += orange_pixels
                total_juxt_area += yellow_pixels
                total_vent_area += blue_pixels
                total_wmh_area += (red_pixels + orange_pixels + yellow_pixels)
                
            except Exception as e:
                print(f"Error processing {npy_file.name}: {e}")
                continue
        
        # Get patient demographics
        patient_demo = self.patient_info.get(patient_id, {
            'age': None,
            'sex': None,
            'group': None
        })
        
        # Store results
        self.results.append({
            'PatientID': patient_id,
            'PatientAge': patient_demo['age'],
            'PatientSex': patient_demo['sex'],
            'StudyGroup': patient_demo['group'],
            'TotalVentricleArea': total_vent_area,
            'TotalWMHArea': total_wmh_area,
            'TotalPeriArea': total_peri_area,
            'TotalParaArea': total_para_area,
            'TotalJuxtArea': total_juxt_area
        })
    
    def save_results(self, output_path, format='csv'):
        """Save results to file"""
        if not self.results:
            print("No results to save!")
            return
        
        df = pd.DataFrame(self.results)
        
        # Sort by PatientID
        df = df.sort_values('PatientID')
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
            print(f"Results saved to {output_path}")
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
            print(f"Results saved to {output_path}")
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total patients processed: {len(df)}")
        if 'StudyGroup' in df.columns:
            group_counts = df['StudyGroup'].value_counts()
            print(f"Study groups: {group_counts.to_dict()}")
        
        print("\nArea statistics (in pixels):")
        numeric_cols = ['TotalVentricleArea', 'TotalWMHArea', 'TotalPeriArea', 'TotalParaArea', 'TotalJuxtArea']
        print(df[numeric_cols].describe())
        
        return df

def main():
    """Main execution function"""
    # Configuration - UPDATE THESE PATHS
    masks_directory = r"E:\MBashiri\Thesis\p4\All_final_np_masks"
    hc_csv_path = r"E:\MBashiri\Thesis\p4\Patient_Flair_data_HC.csv"
    ms_csv_path = r"E:\MBashiri\Thesis\p4\Patient_Flair_data_MS.csv"
    output_path = r"E:\MBashiri\Thesis\p4\brain_mri_analysis_results.csv"
    
    # Initialize extractor
    extractor = BrainMRIDataExtractor(
        masks_directory=masks_directory,
        hc_csv_path=hc_csv_path,
        ms_csv_path=ms_csv_path
    )
    
    try:
        # Load patient demographics
        extractor.load_patient_demographics()
        
        # Process all masks and results
        extractor.process_masks_and_results()
        
        # Save results
        results_df = extractor.save_results(output_path, format='csv')
        
        # Also save as Excel for easier viewing
        excel_path = output_path.replace('.csv', '.xlsx')
        extractor.save_results(excel_path, format='excel')
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"CSV file saved: {output_path}")
        print(f"Excel file saved: {excel_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
