import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y-%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
      level=logging.DEBUG, 
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      )


if __name__ == "__main__":
    logging.debug("This is a debug message")
    