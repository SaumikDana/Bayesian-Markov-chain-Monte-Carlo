import setup_path
from src.imports import *


def numpy_array_encoder(obj):
    """
    Custom JSON encoder function for NumPy arrays.
    
    This function is used as a custom encoder with json.dump() to handle NumPy arrays
    which are not natively JSON serializable. It converts NumPy arrays into a 
    dictionary format that preserves both the data and shape information.
    
    Args:
        obj: The object to be encoded. Expected to be a NumPy array when called
             by the JSON encoder, but will raise TypeError for unsupported types.
    
    Returns:
        dict: A dictionary containing the NumPy array data in the format:
              {
                  "__ndarray__": True,
                  "data": list,  # The array data as a nested list
                  "shape": tuple # The original shape of the array
              }
    
    Raises:
        TypeError: If the object is not a NumPy array or other JSON-serializable type.
    
    Example:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> encoded = numpy_array_encoder(arr)
        >>> print(encoded)
        {'__ndarray__': True, 'data': [[1, 2], [3, 4]], 'shape': (2, 2)}
    
    Note:
        This function is typically used internally by json.dump() and not called directly.
        It's passed as the 'default' parameter to json.dump().
    """
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "shape": obj.shape}
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")


def save_object(obj, filename):
    """
    Saves a Python object to a JSON file with support for NumPy arrays.
    
    This function serializes Python objects to JSON format, with special handling
    for NumPy arrays through the custom numpy_array_encoder. The resulting JSON
    file can be loaded back using the corresponding load_object function.
    
    Args:
        obj: The Python object to save. Can contain nested structures with
             lists, dictionaries, NumPy arrays, and other JSON-serializable types.
        filename (str): The path/filename where the JSON file will be saved.
                       Should include the .json extension for clarity.
    
    Returns:
        None
    
    Raises:
        Exception: Any exception that occurs during file writing or JSON serialization
                  will be caught and printed. Common exceptions include:
                  - FileNotFoundError: Invalid file path
                  - PermissionError: Insufficient write permissions
                  - TypeError: Object contains non-serializable types not handled by encoder
    
    Examples:
        >>> # Save a simple object
        >>> data = {"numbers": [1, 2, 3], "name": "test"}
        >>> save_object(data, "simple_data.json")
        
        >>> # Save object with NumPy arrays
        >>> complex_data = {
        ...     "matrix": np.array([[1, 2], [3, 4]]),
        ...     "vector": np.array([1, 2, 3]),
        ...     "metadata": {"created": "2024"}
        ... }
        >>> save_object(complex_data, "complex_data.json")
    
    Note:
        - The function will overwrite existing files without warning
        - Error messages are printed to stdout but exceptions are suppressed
        - For production use, consider allowing exceptions to propagate or using logging
    """
    try:
        with open(filename, "w") as f:
            json.dump(obj, f, default=numpy_array_encoder)
    except Exception as ex:
        print(f"Error during JSON serialization: {ex}")


def numpy_array_decoder(dct):
    """
    Custom JSON decoder function for reconstructing NumPy arrays.
    
    This function is used as an object hook with json.load() to reconstruct
    NumPy arrays that were encoded using numpy_array_encoder. It identifies
    dictionaries that represent encoded NumPy arrays and converts them back
    to their original NumPy array form.
    
    Args:
        dct (dict): A dictionary from the JSON parsing process. If it contains
                   the special "__ndarray__" key, it will be converted back to
                   a NumPy array. Otherwise, it's returned unchanged.
    
    Returns:
        np.ndarray or dict: Returns a reconstructed NumPy array if the dictionary
                           contains encoded array data, otherwise returns the
                           original dictionary unchanged.
    
    Example:
        >>> # This would typically be called internally by json.load()
        >>> encoded_dict = {
        ...     "__ndarray__": True,
        ...     "data": [[1, 2], [3, 4]],
        ...     "shape": (2, 2)
        ... }
        >>> arr = numpy_array_decoder(encoded_dict)
        >>> print(arr)
        [[1 2]
         [3 4]]
        >>> print(type(arr))
        <class 'numpy.ndarray'>
    
    Note:
        This function is typically used internally by json.load() and not called directly.
        It's passed as the 'object_hook' parameter to json.load().
    """
    if dct.get("__ndarray__"):
        return np.array(dct["data"]).reshape(dct["shape"])
    return dct


def load_object(filename):
    """
    Loads a Python object from a JSON file with support for NumPy arrays.
    
    This function deserializes JSON files back into Python objects, with special
    handling for NumPy arrays that were encoded using the save_object function.
    It can reconstruct complex nested structures containing NumPy arrays.
    
    Args:
        filename (str): The path/filename of the JSON file to load.
                       Should be a file previously created with save_object().
    
    Returns:
        object or None: The deserialized Python object with NumPy arrays properly
                       reconstructed. Returns None if an error occurs during loading.
    
    Raises:
        Exception: Any exception that occurs during file reading or JSON deserialization
                  will be caught and printed. Common exceptions include:
                  - FileNotFoundError: File does not exist
                  - PermissionError: Insufficient read permissions
                  - json.JSONDecodeError: Invalid JSON format
                  - ValueError: Issues during NumPy array reconstruction
    
    Examples:
        >>> # Load a simple object
        >>> data = load_object("simple_data.json")
        >>> print(data)
        {'numbers': [1, 2, 3], 'name': 'test'}
        
        >>> # Load object with NumPy arrays
        >>> complex_data = load_object("complex_data.json")
        >>> print(type(complex_data["matrix"]))
        <class 'numpy.ndarray'>
        >>> print(complex_data["matrix"])
        [[1 2]
         [3 4]]
    
    Note:
        - Returns None on error instead of raising exceptions
        - Error messages are printed to stdout
        - For production use, consider allowing exceptions to propagate or using logging
        - Loaded NumPy arrays will have the same dtype as when they were saved
    """
    try:
        with open(filename, "r") as f:
            return json.load(f, object_hook=numpy_array_decoder)
    except Exception as ex:
        print(f"Error during JSON deserialization: {ex}")