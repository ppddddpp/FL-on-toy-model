from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import os
import pandas as pd
import matplotlib.pyplot as plt
from Helpers.Helpers import log_and_print

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "medical_data" 
EDA_DIR = BASE_DIR / "eda_results"
if not EDA_DIR.exists():
    EDA_DIR.mkdir(parents=True)

def getDataEDA(data_dir=DATA_DIR, save_dir=EDA_DIR, log_file=None):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if log_file is None:
        log_file = save_dir / "dataInfo.log"

    if not csv_files:
        log_and_print("No CSV files found in the specified directory.", log_file=log_file)
        return

    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        # Create a clean name for saving plots
        base_name = Path(file).stem
        
        log_and_print("="*80, log_file=log_file)
        log_and_print(f"Processing file: {file}", log_file=log_file)
        log_and_print("="*80, log_file=log_file)

        try:
            # Read dataset
            df = pd.read_csv(file_path)

            # Overview 
            log_and_print("- Dataset shape:", df.shape, log_file=log_file)
            log_and_print("\n   + Columns:", df.columns.tolist(), log_file=log_file)
            log_and_print("\n   + First 5 rows:", log_file=log_file)
            log_and_print(df.head(), log_file=log_file)

            # Check missing values 
            log_and_print("\n- Missing values per column:", log_file=log_file)
            log_and_print(df.isnull().sum(), log_file=log_file)

            # Extract disease name
            if "Disease_Information" in df.columns:

                # Extract disease name 
                df["Disease_Name"] = df["Disease_Information"].str.split(".").str[0]

                # Count by Body_System
                if "Body_System" in df.columns:
                    system_counts = df["Body_System"].value_counts()
                    log_and_print("\n- Number of diseases per body system:", log_file=log_file)
                    log_and_print(system_counts, log_file=log_file)

                    plt.figure(figsize=(8, 4)) 
                    system_counts.plot(kind="bar", title=f"Diseases per Body System ({file})", log_file=log_file)
                    plt.ylabel("Number of diseases", log_file=log_file)
                    plt.xlabel("Body system", log_file=log_file)
                    plt.tight_layout()
                    
                    # Define the path for saving the figure
                    plot_filename = f"{base_name}_BodySystem_Bar.png"
                    save_path = save_dir / plot_filename
                    
                    # Save the plot
                    plt.savefig(save_path)
                    log_and_print(f"Plot saved to: {save_path}", log_file=log_file)
                    plt.close()

                # Length of description
                df["Description_Length"] = df["Disease_Information"].str.len()
                log_and_print("\n- Description length statistics:", log_file=log_file)
                log_and_print(df["Description_Length"].describe(), log_file=log_file)

                # Check duplicates by Disease_Name
                duplicates = df[df.duplicated("Disease_Name", keep=False)]
                if not duplicates.empty:
                    log_and_print("\n- Duplicate diseases found:", log_file=log_file)
                    log_and_print(duplicates[["Disease_Name", "Body_System"]], log_file=log_file)
                else:
                    log_and_print("\n- No duplicate diseases found.", log_file=log_file)
                log_and_print("\n", log_file=log_file)

        except Exception as e:
            log_and_print(f"Error processing file {file}: {e}", log_file=log_file)

if __name__ == "__main__":
    getDataEDA()