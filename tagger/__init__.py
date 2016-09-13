import configparser
import logging
import os
import sys

from configparser import ConfigParser


config = ConfigParser()
# get config from file
config.read('tagger.ini')

# get config from environment
if not config.has_section('BOTZEN'):
    config.add_section('BOTZEN')
for key, val in os.environ.items():
    if key.lower().startswith('botzen_'):
        # standard interpolation may fail with certain env-values
        try:
            config.set('BOTZEN', key[7:], val)
        except ValueError:
            print("failed to use config value for %s" % (str({key:value})),
                  file=sys.stderr)
            exit(1)

# dict of env-vars
CONFIG_ENV_DEFAULTS = dict(config.items('BOTZEN'))
# name of config section to look for task specific configuration
CONFIG_TASK = config.get('DEFAULT', 'task', vars=CONFIG_ENV_DEFAULTS,
                         fallback='empirist')

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(filename)s,%(funcName)8s(),ln%(lineno) 3s: %(message)s",
        level=getattr(logging,config.get(CONFIG_TASK, 'logging_level',
                      vars=CONFIG_ENV_DEFAULTS)))
logger.debug("config read, logging started...")
