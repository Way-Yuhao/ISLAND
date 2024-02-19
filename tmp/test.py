from util.helper import alert, monitor, timer, monitor_complete
import wandb
import traceback


@timer
@monitor
def stupid_function():
    # 1 / 0
    print('Hello, world!')


if __name__ == "__main__":
    stupid_function()


