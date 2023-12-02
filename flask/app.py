from flask import Flask, jsonify, request
from ..Python.dl_inference import setup_problem, perform_inference

app = Flask(__name__)

@app.route('/setup', methods=['GET'])
def setup():
    # Set up the RSF problem and store the problem instance for later use
    problem = setup_problem()
    # Depending on your implementation, store the problem instance in a session, database, or a file
    # For simplicity, this example will not handle the storage
    return jsonify({"message": "Problem setup complete"})

@app.route('/inference', methods=['GET'])
def inference():
    # Retrieve parameters from the request
    data_format = request.args.get('format', default='json', type=str)
    nsamples = request.args.get('nsamples', default=500, type=int)

    # Retrieve the problem instance from where it was stored
    # This example assumes it's readily available, but you'll need to implement this part
    problem = retrieve_problem()

    if problem is None:
        return jsonify({"error": "Problem not set up"}), 400

    # Perform the inference and return the results
    time_taken = perform_inference(problem, data_format, nsamples)
    return jsonify({"time_taken": time_taken})

# Utility function to retrieve the stored problem instance
# You'll need to implement this according to how you store the problem instance
def retrieve_problem():
    # Retrieve and return the problem instance
    # Placeholder implementation - replace with your actual storage/retrieval mechanism
    return None

if __name__ == '__main__':
    app.run(debug=True)
