import os
import numpy as np
import skimage
import pandas as pd

main_dir = r"E:\MBashiri\Thesis\p4\HC_COHORT_PREP"
csv_file_path = r"E:\MBashiri\ours_articles\Paper#Stats\scripts\brain_mri_analysis_results_PROCESSED.csv"

# Load the CSV file
csv_file = pd.read_csv(csv_file_path)

# Get list of .npz files
brain_info_files = [f for f in os.listdir(main_dir) if f.endswith(".npz")]

# Initialize TotalIntracranialArea column if it doesn't exist
if 'TotalIntracranialArea' not in csv_file.columns:
    csv_file['TotalIntracranialArea'] = np.nan

print(f"Found {len(brain_info_files)} .npz files")
print(f"CSV contains {len(csv_file)} rows")

# Process each brain info file
for brain_info_file in brain_info_files:
    # Extract patient ID from filename (assuming format: PATIENTID_*.npz)
    patient_id = brain_info_file.split('_')[0]

    # Load the .npz file
    brain_info_file_path = os.path.join(main_dir, brain_info_file)

    try:
        brain_info = np.load(brain_info_file_path)

        # Check if required key exists
        if 'brain_mask' not in brain_info:
            print(f"Warning: 'brain_mask' key not found in {brain_info_file}")
            print(f"Available keys: {list(brain_info.keys())}")
            continue

        brain_masks = brain_info['brain_mask']

        # Select slices 7-14 (8th to 15th slice, 0-indexed)
        selected_masks = brain_masks[..., 7:15]

        # Calculate TIA - sum of all mask pixels in selected slices
        # Note: Assuming masks are binary (0s and 1s), so no need to normalize by max
        TIA = np.sum(selected_masks)

        # Alternative calculation if masks are not binary:
        # TIA = np.sum(selected_masks > 0)  # Count non-zero pixels

        print(f"Patient {patient_id}: TIA = {TIA}")

        # Update CSV file - find matching patient ID
        # Convert patient_id to appropriate type (int) to match CSV data type
        try:
            patient_id_numeric = int(patient_id)
        except ValueError:
            print(f"Warning: Could not convert patient ID '{patient_id}' to integer")
            continue

        # Check if column name is "PatinetID" (with typo) or "PatientID"
        patient_id_col = 'PatinetID' if 'PatinetID' in csv_file.columns else 'PatientID'

        if patient_id_col not in csv_file.columns:
            print(f"Warning: Neither 'PatinetID' nor 'PatientID' column found in CSV")
            print(f"Available columns: {list(csv_file.columns)}")
            continue

        # Find matching rows - compare with numeric patient ID
        mask = csv_file[patient_id_col] == patient_id_numeric
        matching_rows = csv_file[mask]

        if len(matching_rows) == 0:
            print(f"Warning: Patient ID {patient_id_numeric} not found in CSV")
        elif len(matching_rows) > 1:
            print(f"Warning: Multiple entries found for Patient ID {patient_id_numeric}")
            # Update all matching rows
            csv_file.loc[mask, 'TotalIntracranialArea'] = TIA
            print(f"Updated TIA for all {len(matching_rows)} entries of Patient {patient_id_numeric}: {TIA}")
        else:
            # Update the single matching row
            csv_file.loc[mask, 'TotalIntracranialArea'] = TIA
            print(f"Updated TIA for Patient {patient_id_numeric}: {TIA}")

    except Exception as e:
        print(f"Error processing {brain_info_file}: {str(e)}")
        continue

# Save the updated CSV file
output_csv_path = csv_file_path.replace('.csv', '_updated.csv')
csv_file.to_csv(output_csv_path, index=False)
print(f"\nUpdated CSV saved to: {output_csv_path}")

# Display summary statistics
tia_values = csv_file['TotalIntracranialArea'].dropna()
if len(tia_values) > 0:
    print(f"\nTIA Summary Statistics:")
    print(f"Count: {len(tia_values)}")
    print(f"Mean: {tia_values.mean():.2f}")
    print(f"Std: {tia_values.std():.2f}")
    print(f"Min: {tia_values.min():.2f}")
    print(f"Max: {tia_values.max():.2f}")
else:
    print("\nNo TIA values calculated successfully.")