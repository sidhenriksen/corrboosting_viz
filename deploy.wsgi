import sys,site,os


site.addsitedir(os.path.join(os.path.dirname(__file__)), 'venv/local/lib64/python3.5/site-packages')
sys.path.append('/var/www/corrboosting_viz')

activate_env = os.path.expanduser(os.path.join(os.path.dirname(__file__),
						'venv/bin/activate_this.py'))
execfile(activate_env, dict(__file__=activate_env))

from app import app as application