from util.helper import capture_stdout

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

# @capture_stdout
def noisy_function(a):
    print("Error This print statement is captured.", a)
    return 42


try:
    yoo = capture_stdout(noisy_function)
    yoo(1)
    # noisy_function()
except ValueError as e:
    # pass
    print(e)
