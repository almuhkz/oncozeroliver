import streamlit as st
import subprocess
import os
import tempfile

# Streamlit app setup
st.title("Liver CT Scan Tumor Prediction")
uploaded_file = st.file_uploader("Upload a CT scan (.nii.gz)", type=["nii.gz"])

def run_validation(file_path, save_dir):
    # Command to run validation.py with appropriate arguments
    cmd = [
        "python", "validation.py",
        "--model", "swin_unetrv2",
        "--swin_type", "base",
        "--val_overlap", "0.75",
        "--val_dir", os.path.dirname(file_path),
        "--json_dir", "datafolds/lits.json",
        "--log_dir", "runs/synt.pretrain.swin_unetrv2_base",
        "--save_dir", save_dir,
        "-W", "ignore"
    ]
    subprocess.run(cmd)

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp.nii.gz")
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        run_validation(temp_file_path, temp_dir)

        prediction_file = os.path.join(temp_dir, "prediction.nii.gz")
        if os.path.exists(prediction_file):
            with open(prediction_file, "rb") as file:
                st.download_button(
                    label="Download Prediction",
                    data=file,
                    file_name="prediction_result.nii.gz",
                    mime="application/gzip"
                )
import sys
print("Python Executable:", sys.executable)

# Reminder: Adjust paths and parameters according to your actual setup
