# Libraries -----------------------------------------------------------------------------------

import os
import sys
sys.path.append(os.getcwd())
import params as params 

# Defining executable file extensions -----------------------------------------------------------------------------------

if params.operating_system == "Windows":
    binary_extensions =".exe"
else:
    binary_extensions = ""

# Info -----------------------------------------------------------------------------------

print(f"-----------------------------------------------------------\nComenzando proceso de entrenamiento\n-----------------------------------------------------------")

# Preprocessing -----------------------------------------------------------------------------------

os.system(f"python{binary_extensions} files/datasets/intermediate/a01_df_all.parquet")