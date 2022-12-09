def beauty_print(content, level=None, type=None):
    if level is None and type is None:
        level = 1
    if level == 0 or type == "warning":
        print("\033[1;31m[Rofunc:WARNING] {}\033[0m".format(content))  # For error and warning (red)
    elif level == 1 or type == "module":
        print("\033[1;33m[Rofunc:MODULE] {}\033[0m".format(content))  # start of a new module (light yellow)
    elif level == 2 or type == "info":
        print("\033[1;35m[Rofunc:INFO] {}\033[0m".format(content))  # start of a new function (light purple)
    elif level == 3:
        print("\033[1;36m{}\033[0m".format(content))  # For mentioning the start of a new class (light cyan)
    elif level == 4:
        print("\033[1;32m{}\033[0m".format(content))  # For mentioning the start of a new method (light green)
    elif level == 5:
        print("\033[1;34m{}\033[0m".format(content))  # For mentioning the start of a new line (light blue)
    else:
        raise ValueError("Invalid level")
