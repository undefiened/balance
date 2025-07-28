"""
File description:
-----------------
This file contains constant values that can be used across the project files.
"""

# --- Time in seconds an ip model is allowed to do solving. ---
#MAXIMUM_RUNTIME = 3*60*60
MAXIMUM_RUNTIME = 3*60*60
# --- If a model reaches this gap percentage, stop further optimization. This is used to speed up solving runtime. ---
ALLOWED_GAP = 0.005
IP_TIME_HORIZON_MULTIPLIER = 3
GREEDY_TIME_HORIZON_MULTIPLIER = 6

USE_WARM_UP_SOLUTION_FOR_IP = True
SORT_INTENTS_BY_DEPARTURE_TIME = False

DEBUG = True
ADD_REVERSE_EDGES = False
# =============================================== END OF FILE ===============================================
