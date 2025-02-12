# Credentials file: scripts/utils/credentials.json
## This is a python class for reading credentials file and extracting the necessary keys.
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarthdataAuth:
    def __init__(self):
        self.earthdata_username = ""
        self.earthdata_password = ""

    def _load_credentials(self, cred_file_path):
        try:
            # Open and read the credentials file
            with open(cred_file_path, "r") as file:
                credentials = json.load(file)
                if credentials:
                    logger.info("Credentials file loaded successfully")
                return credentials
        except Exception as e:
            raise e
    
    def get_earthdata_credentials(self, cred_file_path):
        credentials = self._load_credentials(cred_file_path)
        earthdata_username = credentials["nasa_earthdata_authentication"]["ned_username"]
        earthdata_password = credentials["nasa_earthdata_authentication"]["ned_password"]
        return earthdata_username, earthdata_password