import sys
import time
import math


# Progress bar
class progress_bar():
    """
    A progress bar object will display a progress bar from :
    Args:
      a name
      a size (maximum of possible progress)
      an optinal parameter that will be display at the end of the request
    """
    BAR_SIZE = 40

    def __init__(self, name: str, size: int, parameter: str = ""):
        self.name = name
        self.size = size
        self.parameter = parameter

        self.start = time.time()
        self.update(0)

    def update(self, current_progression: int):
        percent = 100.0 * current_progression / self.size
        sys.stdout.write('\r')
        sys.stdout.write(self.name + " : [{:{}}] {:>3}%"
                         .format('=' * int(percent / (100.0 / self.BAR_SIZE)),
                                 self.BAR_SIZE, int(percent)))
        sys.stdout.write(" " + str(current_progression) + "/" + str(self.size))
        sys.stdout.write(" " + str(self.parameter))
        sys.stdout.write(" " + str(math.floor(time.time() - self.start)) + "s")
        sys.stdout.flush()

        if percent == 100:
            print('')
