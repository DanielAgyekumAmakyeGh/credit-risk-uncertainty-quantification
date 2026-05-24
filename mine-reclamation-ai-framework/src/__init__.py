__version__ = "1.0.0" 
__author__ = "Daniel Agyekum Amakye" 
__license__ = "MIT" 
 
from .phase1_fusion import run_fusion 
from .phase2_prediction import run_prediction 
from .phase3_control import run_control 
 
__all__ = ["run_fusion", "run_prediction", "run_control"] 
