import json
import numpy as np

def numpy_array_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert the NumPy array to a nested list
    raise TypeError("Object of type '{}' is not JSON serializable".format(type(obj)))

def save_object(obj, filename):
    try:
        with open(filename, "w") as f:
            json.dump(obj, f, default=numpy_array_encoder)
    except Exception as ex:
        print("Error during JSON serialization:", ex)

def numpy_array_decoder(dct):
    if "__ndarray__" in dct:
        arr_data = np.asarray(dct["data"])
        return arr_data.reshape(dct["shape"])
    return dct

def load_object(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f, object_hook=numpy_array_decoder)
    except Exception as ex:
        print("Error during JSON deserialization:", ex)
