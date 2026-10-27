"""All file locations in one place. Override with environment variables."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get('LPP_DATA', os.path.join(ROOT, 'data')) + '/'          # raw survey files, see data/README.md
RES = os.environ.get('LPP_RESULTS', os.path.join(ROOT, 'results')) + '/'    # model outputs and evaluation files
PAPER = os.environ.get('LPP_TABLES', os.path.join(ROOT, 'tables')) + '/'    # generated LaTeX tables, macros and figures
os.makedirs(RES, exist_ok=True); os.makedirs(PAPER, exist_ok=True)
