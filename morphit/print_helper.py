print_debug = False


def print_string(string: str):
    if not print_debug:
        return
    print(string)