



from dataclasses import dataclass
from pathlib import Path


#################################### DATA_INGESTION_CONFIG ################################


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    local_data_file : Path



#################################### DATA-VALIDATION-CONFIG ################################


@dataclass(frozen= True)
class DataValidationConfig:
    root_dir : Path
    local_data_file : Path
    STATUS_FILE : Path
    all_schema : Path