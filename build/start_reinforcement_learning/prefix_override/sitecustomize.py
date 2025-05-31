import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aladine/memoir/multi-robot-exploration-rl-master/install/start_reinforcement_learning'
