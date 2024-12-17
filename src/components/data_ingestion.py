import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join('artifacts', "train.csv")
    test_path: str = os.path.join('artifacts', "test.csv")
    raw_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            # Update file path according to actual project structure
            df = pd.read_csv('notebook\data\stud.csv') #notebook\data\stud.csv 
            logging.info("Read the data from the file")

            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)

            logging.info("Train-test split started")

            # Split the data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_df.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_path, index=False, header=True)

            logging.info("Data Ingestion Completed Successfully")

            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    print(f"Data saved successfully:\nTrain Path: {train_path}\nTest Path: {test_path}")
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
