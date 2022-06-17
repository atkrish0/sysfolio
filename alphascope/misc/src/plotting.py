try:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-deep")
except (ModuleNotFoundError, ImportError):
    raise ImportError("Please install matplotlib via pip or poetry")
