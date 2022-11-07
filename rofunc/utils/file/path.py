import rofunc as rf


def get_rofunc_path():
    if not hasattr(rf, "__path__"):
        raise RuntimeError("rofunc package is not installed")
    rofunc_path = list(rf.__path__)[0]
    return rofunc_path
