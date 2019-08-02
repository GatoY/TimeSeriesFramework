import os
import yaml
import json


# ------------------------------------------------------------------------------
def load_config(filepath):
    with open(filepath, 'r') as f:
        config = yaml.load(f)

    print('-------------------------------------------------------------')
    print('configurations: ')
    print(json.dumps(config['config'], indent=2))
    print('-------------------------------------------------------------')
    return config['config']

# ------------------------------------------------------------------------------
# MYSQL
MYSQL_HOST = ''
MYSQL_USER = ''
MYSQL_PASSWORD = ''
MYSQL_DB = ''

# ------------------------------------------------------------------------------
CONFIG = load_config('./config.yml')
ATTRIBUTE= CONFIG['attribute']


# ------------------------------------------------------------------------------
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.system('mkdir {}'.format(DATA_DIR))
