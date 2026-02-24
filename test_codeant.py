def calculate_something(x, y):
    # Intentional minor issues for CodeAnt AI to catch:
    # 1. Missing docstring
    # 2. Unused variable
    # 3. Bare except
    
    unused_var = 100
    
    try:
        result = x / y
        return result
    except:
        pass
    
    return None

if __name__ == "__main__":
    print(calculate_something(10, 2))
