import logging
import colorlog

'''
# --------------------------------------------
# Hongyi Zheng (github: https://github.com/natezhenghy)
# 21/Apr/2021
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''
'''
# --------------------------------------------
# logger
# --------------------------------------------
'''


def logger_info(logger_name: str, log_path: str = 'default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s',
                                      datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        formatter1 = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] : %(message)s',
            log_colors=log_colors_config)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter1)
        sh.setLevel(level)
        log.addHandler(sh)