import os
import sys

def calculate_something(x, y):
    # Intentional minor issues for CodeAnt AI to catch:
    # 1. Missing docstring
    # 2. Unused variable
    # 3. Bare except
    # 4. Complex nested logic
    
    unused_var = 100
    
    if x > 0:
        if y > 0:
            if x + y > 10:
                print("Doing something")
    
    try:
        result = x / y
        return result
    except:
        pass
    
    return None

if __name__ == "__main__":
    # 5. Long line issue
    print(calculate_something(10, 2))
    print("This is a very long string that should exceed the standard line length limit usually enforced by linters like flake8 or black in some configurations.")
