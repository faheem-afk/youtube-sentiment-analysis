from logger_ import logger

class CustomException(Exception):
    def __init__(self, error):
        error_tb = error.__traceback__
        
        error_line_no = error_tb.tb_lineno
        file_name = error_tb.tb_frame.f_code.co_filename
        
        error_message = f"{error} in {file_name} at line number {error_line_no}"
        logger.info(error_message)