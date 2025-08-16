import setup_path
from src.imports import *


def numpy_array_encoder(obj):
    """Custom JSON encoder for NumPy arrays."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "shape": obj.shape}
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")

def save_object(obj, filename):
    """Saves a Python object, including NumPy arrays, to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(obj, f, default=numpy_array_encoder)
    except Exception as ex:
        print(f"Error during JSON serialization: {ex}")

def numpy_array_decoder(dct):
    """Custom JSON decoder for NumPy arrays."""
    if dct.get("__ndarray__"):
        return np.array(dct["data"]).reshape(dct["shape"])
    return dct

def load_object(filename):
    """Loads a Python object, including NumPy arrays, from a JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f, object_hook=numpy_array_decoder)
    except Exception as ex:
        print(f"Error during JSON deserialization: {ex}")
