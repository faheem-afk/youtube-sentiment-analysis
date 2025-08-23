import logging
import os


logger = logging.getLogger()
logger.setLevel('INFO')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

dir = './logs'
file = 'plugin.log'
os.makedirs(dir, exist_ok=True)

file_path = os.path.join(dir, file)

file_handler = logging.FileHandler(file_path)
file_handler.setLevel('INFO')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


