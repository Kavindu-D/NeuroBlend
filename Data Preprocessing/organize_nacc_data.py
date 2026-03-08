# #!/usr/bin/env python3
# """
# NACC Data Organizer - FIXED VERSION
# Organizes matched MRI-PET pairs into structured folders
# """

# import os
# import shutil
# from datetime import datetime
# import re

# # Configuration
# MRI_SOURCE = "/Volumes/BACKUP/MRI"
# PET_SOURCE = "/Volumes/BACKUP/scan/pet"
# OUTPUT_BASE = "/Volumes/BACKUP/NACC_Matched_Pairs2"
# MATCHING_IDS_FILE = "/Users/kavindud/Documents/IRP/nacc_matching_ids.txt"
# LOG_FILE = "data_organization_log.txt"

# def log_message(message, log_file):
#     """Write message to log file and print to console"""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     log_entry = f"[{timestamp}] {message}"
#     print(log_entry)
#     with open(log_file, 'a') as f:
#         f.write(log_entry + "\n")

# def verify_file_copy(source, destination):
#     """Verify that file was copied correctly by comparing sizes"""
#     if not os.path.exists(destination):
#         return False

#     source_size = os.path.getsize(source)
#     dest_size = os.path.getsize(destination)

#     return source_size == dest_size

# def safe_copy_file(source, destination, log_file):
#     """Safely copy file with verification"""
#     try:
#         # Create parent directory if needed
#         os.makedirs(os.path.dirname(destination), exist_ok=True)

#         # Copy file
#         shutil.copy2(source, destination)

#         # Verify copy
#         if verify_file_copy(source, destination):
#             return True
#         else:
#             log_message(f"   ⚠️  WARNING: File size mismatch for {os.path.basename(source)}", log_file)
#             return False

#     except Exception as e:
#         log_message(f"   ❌ ERROR copying {os.path.basename(source)}: {str(e)}", log_file)
#         return False

# def get_files_for_patient(directory, nacc_id):
#     """Get list of files for a specific patient"""
#     try:
#         all_files = os.listdir(directory)
#         pattern = f"SCAN_{nacc_id}_"
#         patient_files = [f for f in all_files 
#                         if f.startswith(pattern) and f.endswith('.zip') and not f.startswith('._')]
#         return patient_files
#     except Exception as e:
#         return []

# def organize_nacc_data():
#     """Main function to organize NACC data"""

#     # Initialize log file
#     log_file = LOG_FILE
#     with open(log_file, 'w') as f:
#         f.write("NACC Data Organization Log - FIXED VERSION\n")
#         f.write("=" * 80 + "\n")
#         f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 80 + "\n\n")

#     log_message("🚀 Starting NACC Data Organization (FIXED VERSION)", log_file)
#     log_message(f"   MRI Source: {MRI_SOURCE}", log_file)
#     log_message(f"   PET Source: {PET_SOURCE}", log_file)
#     log_message(f"   Output Base: {OUTPUT_BASE}", log_file)
#     log_message("", log_file)

#     # Check if matching IDs file exists
#     if not os.path.exists(MATCHING_IDS_FILE):
#         log_message(f"❌ ERROR: Matching IDs file not found: {MATCHING_IDS_FILE}", log_file)
#         log_message("   Please run the bash matching script first!", log_file)
#         return

#     # Check if source directories exist
#     if not os.path.exists(MRI_SOURCE):
#         log_message(f"❌ ERROR: MRI source directory not found: {MRI_SOURCE}", log_file)
#         return

#     if not os.path.exists(PET_SOURCE):
#         log_message(f"❌ ERROR: PET source directory not found: {PET_SOURCE}", log_file)
#         return

#     # Create output base directory
#     log_message("📁 Creating output directory structure...", log_file)
#     try:
#         os.makedirs(OUTPUT_BASE, exist_ok=True)
#         log_message(f"   ✓ Created: {OUTPUT_BASE}", log_file)
#     except Exception as e:
#         log_message(f"   ❌ ERROR: Could not create output directory: {str(e)}", log_file)
#         return

#     log_message("", log_file)

#     # Read matching patient IDs
#     log_message("📋 Reading matching patients list...", log_file)
#     patient_ids = []
#     try:
#         with open(MATCHING_IDS_FILE, 'r') as f:
#             patient_ids = [line.strip() for line in f if line.strip()]
#         log_message(f"   ✓ Found {len(patient_ids)} patients to process", log_file)
#     except Exception as e:
#         log_message(f"   ❌ ERROR reading patient IDs: {str(e)}", log_file)
#         return

#     log_message("", log_file)

#     # Statistics
#     total_patients = len(patient_ids)
#     processed_patients = 0
#     failed_patients = 0
#     total_mri_copied = 0
#     total_pet_copied = 0
#     total_mri_failed = 0
#     total_pet_failed = 0

#     log_message("=" * 80, log_file)
#     log_message("🔄 STARTING FILE ORGANIZATION", log_file)
#     log_message("=" * 80, log_file)
#     log_message("", log_file)

#     # Process each patient
#     for idx, nacc_id in enumerate(patient_ids, 1):

#         # Get files for this patient
#         mri_files = get_files_for_patient(MRI_SOURCE, nacc_id)
#         pet_files = get_files_for_patient(PET_SOURCE, nacc_id)

#         mri_count = len(mri_files)
#         pet_count = len(pet_files)

#         log_message(f"[{idx}/{total_patients}] Processing {nacc_id} ({mri_count} MRI, {pet_count} PET)", log_file)

#         # Skip if no files found (shouldn't happen but safety check)
#         if mri_count == 0 or pet_count == 0:
#             log_message(f"   ⚠️  Skipping: No files found for {nacc_id}", log_file)
#             failed_patients += 1
#             continue

#         # Create patient directory structure
#         patient_dir = os.path.join(OUTPUT_BASE, nacc_id)
#         mri_dir = os.path.join(patient_dir, "MRI")
#         pet_dir = os.path.join(patient_dir, "PET")

#         try:
#             os.makedirs(mri_dir, exist_ok=True)
#             os.makedirs(pet_dir, exist_ok=True)
#         except Exception as e:
#             log_message(f"   ❌ ERROR creating directories for {nacc_id}: {str(e)}", log_file)
#             failed_patients += 1
#             continue

#         # Copy MRI files
#         mri_success = 0
#         for mri_file in mri_files:
#             source = os.path.join(MRI_SOURCE, mri_file)
#             destination = os.path.join(mri_dir, mri_file)

#             if safe_copy_file(source, destination, log_file):
#                 mri_success += 1
#                 total_mri_copied += 1
#             else:
#                 total_mri_failed += 1

#         # Copy PET files
#         pet_success = 0
#         for pet_file in pet_files:
#             source = os.path.join(PET_SOURCE, pet_file)
#             destination = os.path.join(pet_dir, pet_file)

#             if safe_copy_file(source, destination, log_file):
#                 pet_success += 1
#                 total_pet_copied += 1
#             else:
#                 total_pet_failed += 1

#         # Log patient results
#         if mri_success == mri_count and pet_success == pet_count:
#             log_message(f"   ✓ Successfully copied {mri_success} MRI + {pet_success} PET files", log_file)
#             processed_patients += 1
#         else:
#             log_message(f"   ⚠️  Partial copy: {mri_success}/{mri_count} MRI, {pet_success}/{pet_count} PET", log_file)
#             if mri_success > 0 or pet_success > 0:
#                 processed_patients += 1
#             else:
#                 failed_patients += 1

#         # Progress update every 50 patients
#         if idx % 50 == 0:
#             log_message("", log_file)
#             log_message(f"   📊 Progress: {idx}/{total_patients} patients ({(idx/total_patients)*100:.1f}%)", log_file)
#             log_message(f"   📊 Files copied so far: {total_mri_copied} MRI + {total_pet_copied} PET", log_file)
#             log_message("", log_file)

#     # Final summary
#     log_message("", log_file)
#     log_message("=" * 80, log_file)
#     log_message("✅ DATA ORGANIZATION COMPLETE", log_file)
#     log_message("=" * 80, log_file)
#     log_message("", log_file)
#     log_message("📊 FINAL STATISTICS:", log_file)
#     log_message(f"   • Total patients processed: {processed_patients}/{total_patients}", log_file)
#     log_message(f"   • Patients with issues: {failed_patients}", log_file)
#     log_message(f"   • Total MRI files copied: {total_mri_copied}", log_file)
#     log_message(f"   • Total PET files copied: {total_pet_copied}", log_file)
#     log_message(f"   • Total files copied: {total_mri_copied + total_pet_copied}", log_file)

#     if total_mri_failed > 0 or total_pet_failed > 0:
#         log_message("", log_file)
#         log_message("⚠️  WARNINGS:", log_file)
#         log_message(f"   • Failed MRI copies: {total_mri_failed}", log_file)
#         log_message(f"   • Failed PET copies: {total_pet_failed}", log_file)

#     log_message("", log_file)
#     log_message(f"📁 Output location: {OUTPUT_BASE}", log_file)
#     log_message(f"📄 Log file: {log_file}", log_file)
#     log_message("", log_file)
#     log_message(f"⏱️  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
#     log_message("=" * 80, log_file)

#     print("\n✓ All done! Check the log file for details.")

# if __name__ == "__main__":
#     organize_nacc_data()



# RESUMING FIXED
#!/usr/bin/env python3
"""
NACC Data Organizer - RESUME VERSION
Skips already-completed patients, only processes what's missing
"""

import os
import shutil
from datetime import datetime

# Configuration
MRI_SOURCE = "/Volumes/BACKUP/MRI"
PET_SOURCE = "/Volumes/BACKUP/scan/pet"
OUTPUT_BASE = "/Volumes/BACKUP/NACC_Matched_Pairs"
MATCHING_IDS_FILE = "nacc_matching_ids.txt"
LOG_FILE = "data_organization_log_resume.txt"

def log_message(message, log_file):
    """Write message to log file and print to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a') as f:
        f.write(log_entry + "\n")

def verify_file_copy(source, destination):
    """Verify that file was copied correctly by comparing sizes"""
    if not os.path.exists(destination):
        return False

    source_size = os.path.getsize(source)
    dest_size = os.path.getsize(destination)

    return source_size == dest_size

def safe_copy_file(source, destination, log_file):
    """Safely copy file with verification"""
    try:
        # Create parent directory if needed
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Copy file
        shutil.copy2(source, destination)

        # Verify copy
        if verify_file_copy(source, destination):
            return True
        else:
            log_message(f"   ⚠️  WARNING: File size mismatch for {os.path.basename(source)}", log_file)
            return False

    except Exception as e:
        log_message(f"   ❌ ERROR copying {os.path.basename(source)}: {str(e)}", log_file)
        return False

def get_files_for_patient(directory, nacc_id):
    """Get list of files for a specific patient"""
    try:
        all_files = os.listdir(directory)
        pattern = f"SCAN_{nacc_id}_"
        patient_files = [f for f in all_files 
                        if f.startswith(pattern) and f.endswith('.zip') and not f.startswith('._')]
        return patient_files
    except Exception as e:
        return []

def check_patient_complete(patient_dir, expected_mri, expected_pet):
    """Check if patient folder already has correct number of files"""
    mri_dir = os.path.join(patient_dir, "MRI")
    pet_dir = os.path.join(patient_dir, "PET")

    if not os.path.exists(mri_dir) or not os.path.exists(pet_dir):
        return False

    try:
        mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.zip') and not f.startswith('._')]
        pet_files = [f for f in os.listdir(pet_dir) if f.endswith('.zip') and not f.startswith('._')]

        return len(mri_files) == expected_mri and len(pet_files) == expected_pet
    except:
        return False

def organize_nacc_data():
    """Main function to organize NACC data"""

    # Initialize log file
    log_file = LOG_FILE
    with open(log_file, 'w') as f:
        f.write("NACC Data Organization Log - RESUME VERSION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    log_message("🚀 Starting NACC Data Organization (RESUME VERSION - skips completed)", log_file)
    log_message(f"   MRI Source: {MRI_SOURCE}", log_file)
    log_message(f"   PET Source: {PET_SOURCE}", log_file)
    log_message(f"   Output Base: {OUTPUT_BASE}", log_file)
    log_message("", log_file)

    # Check if matching IDs file exists
    if not os.path.exists(MATCHING_IDS_FILE):
        log_message(f"❌ ERROR: Matching IDs file not found: {MATCHING_IDS_FILE}", log_file)
        return

    # Check if source directories exist
    if not os.path.exists(MRI_SOURCE):
        log_message(f"❌ ERROR: MRI source directory not found: {MRI_SOURCE}", log_file)
        return

    if not os.path.exists(PET_SOURCE):
        log_message(f"❌ ERROR: PET source directory not found: {PET_SOURCE}", log_file)
        return

    # Create output base directory
    log_message("📁 Checking output directory...", log_file)
    try:
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        log_message(f"   ✓ Output directory ready: {OUTPUT_BASE}", log_file)
    except Exception as e:
        log_message(f"   ❌ ERROR: Could not create output directory: {str(e)}", log_file)
        return

    log_message("", log_file)

    # Read matching patient IDs
    log_message("📋 Reading matching patients list...", log_file)
    patient_ids = []
    try:
        with open(MATCHING_IDS_FILE, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        log_message(f"   ✓ Found {len(patient_ids)} patients in list", log_file)
    except Exception as e:
        log_message(f"   ❌ ERROR reading patient IDs: {str(e)}", log_file)
        return

    log_message("", log_file)

    # Statistics
    total_patients = len(patient_ids)
    skipped_complete = 0
    processed_patients = 0
    failed_patients = 0
    total_mri_copied = 0
    total_pet_copied = 0

    log_message("=" * 80, log_file)
    log_message("🔄 STARTING FILE ORGANIZATION (RESUME MODE)", log_file)
    log_message("=" * 80, log_file)
    log_message("", log_file)

    # Process each patient
    for idx, nacc_id in enumerate(patient_ids, 1):

        # Get files for this patient
        mri_files = get_files_for_patient(MRI_SOURCE, nacc_id)
        pet_files = get_files_for_patient(PET_SOURCE, nacc_id)

        mri_count = len(mri_files)
        pet_count = len(pet_files)

        # Skip if no files found
        if mri_count == 0 or pet_count == 0:
            if idx % 100 == 0:  # Only log every 100 skipped
                log_message(f"[{idx}/{total_patients}] Skipping {nacc_id}: No source files found", log_file)
            failed_patients += 1
            continue

        # Check if already complete
        patient_dir = os.path.join(OUTPUT_BASE, nacc_id)
        if check_patient_complete(patient_dir, mri_count, pet_count):
            skipped_complete += 1
            if idx % 100 == 0:  # Only log every 100 skipped
                log_message(f"[{idx}/{total_patients}] Skipping {nacc_id}: Already complete ({mri_count} MRI, {pet_count} PET)", log_file)
            continue

        log_message(f"[{idx}/{total_patients}] Processing {nacc_id} ({mri_count} MRI, {pet_count} PET)", log_file)

        # Create patient directory structure
        mri_dir = os.path.join(patient_dir, "MRI")
        pet_dir = os.path.join(patient_dir, "PET")

        try:
            os.makedirs(mri_dir, exist_ok=True)
            os.makedirs(pet_dir, exist_ok=True)
        except Exception as e:
            log_message(f"   ❌ ERROR creating directories for {nacc_id}: {str(e)}", log_file)
            failed_patients += 1
            continue

        # Copy MRI files (skip if already exists with correct size)
        mri_success = 0
        for mri_file in mri_files:
            source = os.path.join(MRI_SOURCE, mri_file)
            destination = os.path.join(mri_dir, mri_file)

            # Check if file already exists with correct size
            if os.path.exists(destination) and verify_file_copy(source, destination):
                mri_success += 1
                continue

            if safe_copy_file(source, destination, log_file):
                mri_success += 1
                total_mri_copied += 1

        # Copy PET files (skip if already exists with correct size)
        pet_success = 0
        for pet_file in pet_files:
            source = os.path.join(PET_SOURCE, pet_file)
            destination = os.path.join(pet_dir, pet_file)

            # Check if file already exists with correct size
            if os.path.exists(destination) and verify_file_copy(source, destination):
                pet_success += 1
                continue

            if safe_copy_file(source, destination, log_file):
                pet_success += 1
                total_pet_copied += 1

        # Log patient results
        if mri_success == mri_count and pet_success == pet_count:
            log_message(f"   ✓ Complete: {mri_success} MRI + {pet_success} PET files", log_file)
            processed_patients += 1
        else:
            log_message(f"   ⚠️  Partial: {mri_success}/{mri_count} MRI, {pet_success}/{pet_count} PET", log_file)
            processed_patients += 1

        # Progress update every 50 NEW patients (not skipped)
        if processed_patients % 50 == 0:
            log_message("", log_file)
            log_message(f"   📊 Progress: {idx}/{total_patients} checked, {processed_patients} processed, {skipped_complete} already complete", log_file)
            log_message(f"   📊 New files copied: {total_mri_copied} MRI + {total_pet_copied} PET", log_file)
            log_message("", log_file)

    # Final summary
    log_message("", log_file)
    log_message("=" * 80, log_file)
    log_message("✅ DATA ORGANIZATION COMPLETE", log_file)
    log_message("=" * 80, log_file)
    log_message("", log_file)
    log_message("📊 FINAL STATISTICS:", log_file)
    log_message(f"   • Total patients in list: {total_patients}", log_file)
    log_message(f"   • Already complete (skipped): {skipped_complete}", log_file)
    log_message(f"   • Newly processed: {processed_patients}", log_file)
    log_message(f"   • No source files found: {failed_patients}", log_file)
    log_message(f"   • New MRI files copied: {total_mri_copied}", log_file)
    log_message(f"   • New PET files copied: {total_pet_copied}", log_file)
    log_message(f"   • Total new files copied: {total_mri_copied + total_pet_copied}", log_file)
    log_message("", log_file)
    log_message(f"📁 Output location: {OUTPUT_BASE}", log_file)
    log_message(f"📄 Log file: {log_file}", log_file)
    log_message("", log_file)
    log_message(f"⏱️  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_message("=" * 80, log_file)

    print("\n✓ All done! Check the log file for details.")

if __name__ == "__main__":
    organize_nacc_data()#!/usr/bin/env python3
"""
NACC Data Organizer - RESUME VERSION
Skips already-completed patients, only processes what's missing
"""

import os
import shutil
from datetime import datetime

# Configuration
MRI_SOURCE = "/Volumes/BACKUP/MRI"
PET_SOURCE = "/Volumes/BACKUP/scan/pet"
OUTPUT_BASE = "/Volumes/BACKUP/NACC_Matched_Pairs"
MATCHING_IDS_FILE = "nacc_matching_ids.txt"
LOG_FILE = "data_organization_log_resume.txt"

def log_message(message, log_file):
    """Write message to log file and print to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a') as f:
        f.write(log_entry + "\n")

def verify_file_copy(source, destination):
    """Verify that file was copied correctly by comparing sizes"""
    if not os.path.exists(destination):
        return False

    source_size = os.path.getsize(source)
    dest_size = os.path.getsize(destination)

    return source_size == dest_size

def safe_copy_file(source, destination, log_file):
    """Safely copy file with verification"""
    try:
        # Create parent directory if needed
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Copy file
        shutil.copy2(source, destination)

        # Verify copy
        if verify_file_copy(source, destination):
            return True
        else:
            log_message(f"   ⚠️  WARNING: File size mismatch for {os.path.basename(source)}", log_file)
            return False

    except Exception as e:
        log_message(f"   ❌ ERROR copying {os.path.basename(source)}: {str(e)}", log_file)
        return False

def get_files_for_patient(directory, nacc_id):
    """Get list of files for a specific patient"""
    try:
        all_files = os.listdir(directory)
        pattern = f"SCAN_{nacc_id}_"
        patient_files = [f for f in all_files 
                        if f.startswith(pattern) and f.endswith('.zip') and not f.startswith('._')]
        return patient_files
    except Exception as e:
        return []

def check_patient_complete(patient_dir, expected_mri, expected_pet):
    """Check if patient folder already has correct number of files"""
    mri_dir = os.path.join(patient_dir, "MRI")
    pet_dir = os.path.join(patient_dir, "PET")

    if not os.path.exists(mri_dir) or not os.path.exists(pet_dir):
        return False

    try:
        mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.zip') and not f.startswith('._')]
        pet_files = [f for f in os.listdir(pet_dir) if f.endswith('.zip') and not f.startswith('._')]

        return len(mri_files) == expected_mri and len(pet_files) == expected_pet
    except:
        return False

def organize_nacc_data():
    """Main function to organize NACC data"""

    # Initialize log file
    log_file = LOG_FILE
    with open(log_file, 'w') as f:
        f.write("NACC Data Organization Log - RESUME VERSION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    log_message("🚀 Starting NACC Data Organization (RESUME VERSION - skips completed)", log_file)
    log_message(f"   MRI Source: {MRI_SOURCE}", log_file)
    log_message(f"   PET Source: {PET_SOURCE}", log_file)
    log_message(f"   Output Base: {OUTPUT_BASE}", log_file)
    log_message("", log_file)

    # Check if matching IDs file exists
    if not os.path.exists(MATCHING_IDS_FILE):
        log_message(f"❌ ERROR: Matching IDs file not found: {MATCHING_IDS_FILE}", log_file)
        return

    # Check if source directories exist
    if not os.path.exists(MRI_SOURCE):
        log_message(f"❌ ERROR: MRI source directory not found: {MRI_SOURCE}", log_file)
        return

    if not os.path.exists(PET_SOURCE):
        log_message(f"❌ ERROR: PET source directory not found: {PET_SOURCE}", log_file)
        return

    # Create output base directory
    log_message("📁 Checking output directory...", log_file)
    try:
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        log_message(f"   ✓ Output directory ready: {OUTPUT_BASE}", log_file)
    except Exception as e:
        log_message(f"   ❌ ERROR: Could not create output directory: {str(e)}", log_file)
        return

    log_message("", log_file)

    # Read matching patient IDs
    log_message("📋 Reading matching patients list...", log_file)
    patient_ids = []
    try:
        with open(MATCHING_IDS_FILE, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        log_message(f"   ✓ Found {len(patient_ids)} patients in list", log_file)
    except Exception as e:
        log_message(f"   ❌ ERROR reading patient IDs: {str(e)}", log_file)
        return

    log_message("", log_file)

    # Statistics
    total_patients = len(patient_ids)
    skipped_complete = 0
    processed_patients = 0
    failed_patients = 0
    total_mri_copied = 0
    total_pet_copied = 0

    log_message("=" * 80, log_file)
    log_message("🔄 STARTING FILE ORGANIZATION (RESUME MODE)", log_file)
    log_message("=" * 80, log_file)
    log_message("", log_file)

    # Process each patient
    for idx, nacc_id in enumerate(patient_ids, 1):

        # Get files for this patient
        mri_files = get_files_for_patient(MRI_SOURCE, nacc_id)
        pet_files = get_files_for_patient(PET_SOURCE, nacc_id)

        mri_count = len(mri_files)
        pet_count = len(pet_files)

        # Skip if no files found
        if mri_count == 0 or pet_count == 0:
            if idx % 100 == 0:  # Only log every 100 skipped
                log_message(f"[{idx}/{total_patients}] Skipping {nacc_id}: No source files found", log_file)
            failed_patients += 1
            continue

        # Check if already complete
        patient_dir = os.path.join(OUTPUT_BASE, nacc_id)
        if check_patient_complete(patient_dir, mri_count, pet_count):
            skipped_complete += 1
            if idx % 100 == 0:  # Only log every 100 skipped
                log_message(f"[{idx}/{total_patients}] Skipping {nacc_id}: Already complete ({mri_count} MRI, {pet_count} PET)", log_file)
            continue

        log_message(f"[{idx}/{total_patients}] Processing {nacc_id} ({mri_count} MRI, {pet_count} PET)", log_file)

        # Create patient directory structure
        mri_dir = os.path.join(patient_dir, "MRI")
        pet_dir = os.path.join(patient_dir, "PET")

        try:
            os.makedirs(mri_dir, exist_ok=True)
            os.makedirs(pet_dir, exist_ok=True)
        except Exception as e:
            log_message(f"   ❌ ERROR creating directories for {nacc_id}: {str(e)}", log_file)
            failed_patients += 1
            continue

        # Copy MRI files (skip if already exists with correct size)
        mri_success = 0
        for mri_file in mri_files:
            source = os.path.join(MRI_SOURCE, mri_file)
            destination = os.path.join(mri_dir, mri_file)

            # Check if file already exists with correct size
            if os.path.exists(destination) and verify_file_copy(source, destination):
                mri_success += 1
                continue

            if safe_copy_file(source, destination, log_file):
                mri_success += 1
                total_mri_copied += 1

        # Copy PET files (skip if already exists with correct size)
        pet_success = 0
        for pet_file in pet_files:
            source = os.path.join(PET_SOURCE, pet_file)
            destination = os.path.join(pet_dir, pet_file)

            # Check if file already exists with correct size
            if os.path.exists(destination) and verify_file_copy(source, destination):
                pet_success += 1
                continue

            if safe_copy_file(source, destination, log_file):
                pet_success += 1
                total_pet_copied += 1

        # Log patient results
        if mri_success == mri_count and pet_success == pet_count:
            log_message(f"   ✓ Complete: {mri_success} MRI + {pet_success} PET files", log_file)
            processed_patients += 1
        else:
            log_message(f"   ⚠️  Partial: {mri_success}/{mri_count} MRI, {pet_success}/{pet_count} PET", log_file)
            processed_patients += 1

        # Progress update every 50 NEW patients (not skipped)
        if processed_patients % 50 == 0:
            log_message("", log_file)
            log_message(f"   📊 Progress: {idx}/{total_patients} checked, {processed_patients} processed, {skipped_complete} already complete", log_file)
            log_message(f"   📊 New files copied: {total_mri_copied} MRI + {total_pet_copied} PET", log_file)
            log_message("", log_file)

    # Final summary
    log_message("", log_file)
    log_message("=" * 80, log_file)
    log_message("✅ DATA ORGANIZATION COMPLETE", log_file)
    log_message("=" * 80, log_file)
    log_message("", log_file)
    log_message("📊 FINAL STATISTICS:", log_file)
    log_message(f"   • Total patients in list: {total_patients}", log_file)
    log_message(f"   • Already complete (skipped): {skipped_complete}", log_file)
    log_message(f"   • Newly processed: {processed_patients}", log_file)
    log_message(f"   • No source files found: {failed_patients}", log_file)
    log_message(f"   • New MRI files copied: {total_mri_copied}", log_file)
    log_message(f"   • New PET files copied: {total_pet_copied}", log_file)
    log_message(f"   • Total new files copied: {total_mri_copied + total_pet_copied}", log_file)
    log_message("", log_file)
    log_message(f"📁 Output location: {OUTPUT_BASE}", log_file)
    log_message(f"📄 Log file: {log_file}", log_file)
    log_message("", log_file)
    log_message(f"⏱️  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_message("=" * 80, log_file)

    print("\n✓ All done! Check the log file for details.")

if __name__ == "__main__":
    organize_nacc_data()