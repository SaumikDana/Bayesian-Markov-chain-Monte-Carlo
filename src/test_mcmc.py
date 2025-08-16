import setup_path
from src.imports import *


class MockModel:
    def evaluate(self):
        # Mock implementation
        return np.random.rand(10), np.random.rand(10)

class TestMCMC:
    @pytest.fixture
    def mock_model(self):
        return MockModel()

    @pytest.fixture
    def mock_data(self):
        return np.random.rand(100)  # Adjust size as needed

    @pytest.fixture
    def mcmc_instance(self, mock_model, mock_data):
        qpriors = [1.0, 0.0, 10000.0]  # Adjust values as needed
        qstart = 1000.0  # Adjust qstart as needed
        dc_true = 500.0  # Adjust dc_true as needed
        return MCMC(mock_model, mock_data, dc_true, qpriors, qstart)

    def test_evaluate_model(self, mcmc_instance):
        # Example test for the evaluate_model method
        result = mcmc_instance.evaluate_model()
        assert isinstance(result, np.ndarray), "evaluate_model should return a numpy array"
