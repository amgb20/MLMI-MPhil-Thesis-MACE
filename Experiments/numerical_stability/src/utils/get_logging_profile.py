from datetime import date, datetime
import logging
import os

# Define the date and time format for the filename
datetimefmt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Get the correct path relative to the current working directory
log_dir = os.path.join('logs', 'cli_run_logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
   level=logging.INFO,
   format="%(levelname)s:%(asctime)s %(message)s",
   datefmt="%Y-%m-%d %H:%M:%S",
   handlers=[
      logging.FileHandler(filename=os.path.join(log_dir, f'logging_profile_{datetimefmt}.log')),
      logging.StreamHandler()
   ]
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Export the logger for use in other modules
__all__ = ['logger']