



from dataclasses import dataclass
from pathlib import Path


#################################### DATA_INGESTION_CONFIG ################################


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    local_data_file : Path