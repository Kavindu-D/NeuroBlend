#!/usr/bin/env python3
"""
NACC Quick Test Preprocessing - ALL SERIES VERSION

For each of the first N patients:
  - Extract all ZIPs (if not already extracted)
  - Convert all DICOM series to NIfTI via dcm2niix
  - Preprocess ALL MRI and PET NIfTI volumes
  - Save each preprocessed volume as a separate .npy file per series

Output structure (example):
/Volumes/BACKUP/NACC_Test_Preprocessed_All/
  NACC000133/
    MRI/
      series_000/ mri_series_000.npy
      series_001/ mri_series_001.npy
    PET/
      series_000/ pet_series_000.npy
      series_001/ pet_series_001.npy
"""

import os
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import nibabel as nib

class AllSeriesPreprocessor:
    def __init__(self, num_patients=50,
                 input_base="/Volumes/BACKUP/NACC_Matched_Pairs",
                 output_base="/Volumes/BACKUP/NACC_Test_Preprocessed_All",
                 log_file="quick_test_all_series_log.txt"):
        self.num_patients = num_patients
        self.input_base = Path(input_base)
        self.output_base = Path(output_base)
        self.log_file = log_file
        self._setup_logging()
        # Explicit dcm2niix path known to work on your Mac
        self.dcm2niix_cmd = Path("/usr/local/bin/dcm2niix")

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__ + "_all_series")
        self.logger.setLevel(logging.INFO)
        # prevent duplicate handlers if re-run in same interpreter
        if self.logger.handlers:
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)
        fh = logging.FileHandler(self.log_file)
        ch = logging.StreamHandler()
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log(self, msg):
        self.logger.info(msg)

    # ---------------- PHASE 1: ZIP extraction ----------------
    def extract_zips(self, patients):
        self.log("="*90)
        self.log("STEP 1/3: EXTRACTING ALL ZIP FILES (IF NEEDED)")
        self.log("="*90)
        extracted = 0
        for idx, pid in enumerate(patients, 1):
            pdir = self.input_base / pid
            for mod in ("MRI", "PET"):
                mdir = pdir / mod
                if not mdir.is_dir():
                    continue
                for z in sorted(mdir.glob('*.zip')):
                    out_dir = mdir / z.stem
                    if out_dir.is_dir():
                        continue
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(z, 'r') as zf:
                            zf.extractall(out_dir)
                        extracted += 1
                    except Exception as e:
                        self.log(f"ERROR extracting {pid}/{mod}/{z.name}: {e}")
            if idx % 10 == 0:
                self.log(f"[{idx}/{len(patients)}] Extraction progress: {extracted} zip folders extracted")
        self.log(f"✓ Step 1 complete: {extracted} zip folders extracted (new)")
        self.log("")

    # ---------------- PHASE 2: DICOM → NIfTI ----------------
    def convert_all_dicoms(self, patients):
        self.log("="*90)
        self.log("STEP 2/3: CONVERTING ALL DICOM SERIES TO NIFTI")
        self.log("="*90)
        # simple check: binary exists
        if not self.dcm2niix_cmd.is_file():
            self.log(f"❌ dcm2niix not found at {self.dcm2niix_cmd}")
            self.log("Install with: brew install dcm2niix")
            return 0
        conversions = 0
        for idx, pid in enumerate(patients, 1):
            pdir = self.input_base / pid
            for mod in ("MRI", "PET"):
                mdir = pdir / mod
                if not mdir.is_dir():
                    continue
                # each extracted zip directory is a series folder
                for series_dir in sorted([d for d in mdir.iterdir() if d.is_dir()]):
                    # Skip if NIfTI already present
                    has_nifti = any(f.suffix in ('.nii', '.gz') and f.name.endswith(('.nii','.nii.gz'))
                                     for f in series_dir.iterdir() if f.is_file())
                    if has_nifti:
                        continue
                    try:
                        # run dcm2niix on this series directory
                        cmd = [str(self.dcm2niix_cmd), '-f', series_dir.name, '-o', str(series_dir), str(series_dir)]
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if r.returncode != 0:
                            self.log(f"WARN: dcm2niix non-zero for {pid}/{mod}/{series_dir.name}: rc={r.returncode}")
                            self.log((r.stderr or r.stdout).strip())
                        # count any nifti created
                        if any(f.name.endswith(('.nii', '.nii.gz')) for f in series_dir.iterdir() if f.is_file()):
                            conversions += 1
                    except Exception as e:
                        self.log(f"ERROR converting {pid}/{mod}/{series_dir.name}: {e}")
            if idx % 10 == 0:
                self.log(f"[{idx}/{len(patients)}] Conversion progress: {conversions} series with NIfTI")
        self.log(f"✓ Step 2 complete: {conversions} series converted to NIfTI (new)")
        self.log("")
        return conversions

    # ---------------- PHASE 3: Preprocess all NIfTI ----------------
    def preprocess_all_series(self, patients):
        self.log("="*90)
        self.log("STEP 3/3: PREPROCESSING ALL MRI & PET SERIES")
        self.log("="*90)
        self.output_base.mkdir(parents=True, exist_ok=True)
        processed = 0
        for idx, pid in enumerate(patients, 1):
            in_patient = self.input_base / pid
            out_patient = self.output_base / pid
            out_patient.mkdir(parents=True, exist_ok=True)
            for mod in ("MRI", "PET"):
                in_mod = in_patient / mod
                if not in_mod.is_dir():
                    continue
                out_mod = out_patient / mod
                out_mod.mkdir(parents=True, exist_ok=True)
                # iterate over each series folder
                for s_idx, series_dir in enumerate(sorted([d for d in in_mod.iterdir() if d.is_dir()])):
                    # each series_dir may contain one or more NIfTI files
                    nifti_files = [f for f in series_dir.iterdir() if f.is_file() and f.name.endswith(('.nii','.nii.gz'))]
                    if not nifti_files:
                        continue
                    # For now, process the first NIfTI per series
                    nifti_path = nifti_files[0]
                    try:
                        img = nib.load(str(nifti_path))
                        data = img.get_fdata()
                        data[data < 0] = 0
                        nonzero = data[data > 0]
                        if nonzero.size < 100:
                            continue
                        if mod == "MRI":
                            v_min = float(np.percentile(nonzero, 0.5))
                            v_max = float(np.percentile(nonzero, 99.5))
                        else:  # PET
                            v_min = float(np.percentile(nonzero, 1.0))
                            v_max = float(np.percentile(nonzero, 99.0))
                        clipped = np.clip(data, v_min, v_max)
                        norm01 = (clipped - v_min) / (v_max - v_min + 1e-8)
                        norm11 = 2*norm01 - 1
                        # save per-series file
                        series_out_dir = out_mod / f"series_{s_idx:03d}"
                        series_out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = series_out_dir / ("mri_series.npy" if mod=="MRI" else "pet_series.npy")
                        np.save(out_path, norm11.astype('float32'))
                        processed += 1
                    except Exception as e:
                        self.log(f"ERROR preprocessing {pid}/{mod}/{series_dir.name}: {e}")
            if idx % 10 == 0:
                self.log(f"[{idx}/{len(patients)}] Preprocessing progress: {processed} series saved")
        self.log(f"✓ Step 3 complete: {processed} series preprocessed and saved as .npy")
        self.log("")
        return processed

    def run(self):
        self.log("="*90)
        self.log("🚀 ALL-SERIES PREPROCESSING PIPELINE (QUICK TEST)")
        self.log("="*90)
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # get patients
        all_pats = sorted([d.name for d in self.input_base.iterdir() if d.is_dir()])
        test_pats = all_pats[:self.num_patients]
        self.log(f"Total patients available: {len(all_pats)}")
        self.log(f"Patients to process: {len(test_pats)} (first {self.num_patients})")
        self.log("")
        # run phases
        self.extract_zips(test_pats)
        conv = self.convert_all_dicoms(test_pats)
        proc = self.preprocess_all_series(test_pats)
        self.log("="*90)
        self.log("✅ ALL-SERIES QUICK TEST COMPLETE")
        self.log("="*90)
        self.log(f"Summary:")
        self.log(f"  • Patients processed: {len(test_pats)}")
        self.log(f"  • Series with NIfTI created (or already present): {conv}")
        self.log(f"  • Preprocessed series saved: {proc}")
        self.log(f"Output root: {self.output_base}")

if __name__ == '__main__':
    p = AllSeriesPreprocessor(num_patients=50)
    p.run()
