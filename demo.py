'''from src.logger import logging
#testing logger
logging.debug('Working fine')
logging.error('Error while opening a file')
logging.critical('Data Not fetched')'''


'''from src.logger import logging
from src.exception import MyException
import sys

try:
    a = 1+'Z'
except Exception as e:
     logging.info(e)
     raise MyException(e, sys) from e'''
     
from src.pipline.main_pipeline import TrainPipeline

pipline = TrainPipeline()
pipline.run_pipeline()