from datetime import date, datetime
import logging

# Define the date and time format for the filename
datetimefmt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
   level=logging.INFO,
   format="%(levelname)s:%(asctime)s %(message)s",
   datefmt="%Y-%m-%d %H:%M:%S",
   handlers=[
      logging.FileHandler(filename=f'Experiments/numerical_stability/cli_run_logs/logging_profile_{datetimefmt}.log'),
      logging.StreamHandler()
   ]
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Export the logger for use in other modules
__all__ = ['logger']