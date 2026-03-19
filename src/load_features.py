import numpy as np

def load_features():
    X_image = np.load("notebooks/X_image.npy")
    X_meta = np.load("notebooks/X_meta.npy")
    y = np.load("notebooks/y.npy")

    return X_image, X_meta, y


if __name__ == "__main__":
    X_image, X_meta, y = load_features()

    print("X_image:", X_image.shape)
    print("X_meta:", X_meta.shape)
    print("y:", y.shape)
