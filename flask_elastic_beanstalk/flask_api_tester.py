import requests

def main():
    # For RunSimulation
    simulation_url = 'http://localhost:5000/run-simulation'
    simulation_data = {"nslips": 5, "lowest": 100.0, "largest": 1000.0}
    try:
        simulation_response = requests.post(simulation_url, json=simulation_data)
        if simulation_response.status_code == 200:
            print("Simulation Response:", simulation_response.json())
        else:
            print("Error with Simulation request:", simulation_response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error during Simulation request:", e)

    # For RunInference
    inference_url = 'http://localhost:5000/run-inference'
    inference_data = {"nsamples": 500}
    try:
        inference_response = requests.post(inference_url, json=inference_data)
        if inference_response.status_code == 200:
            print("Inference Response:", inference_response.json())
        else:
            print("Error with Inference request:", inference_response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error during Inference request:", e)

if __name__ == "__main__":
    main()
