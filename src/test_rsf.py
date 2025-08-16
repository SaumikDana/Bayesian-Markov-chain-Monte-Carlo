import setup_path
from src.imports import *


class TestRSF:
    @pytest.fixture
    def mock_model(self):
        # Mock RateStateModel class
        class MockRateStateModel:
            num_tsteps = 500
            t_start = 0.0
            t_final = 50.0
            Dc = 500.0

            def evaluate(self):
                # Mock implementation
                return np.linspace(self.t_start, self.t_final, self.num_tsteps), np.random.rand(self.num_tsteps), np.random.rand(self.num_tsteps)

        return MockRateStateModel()

    @pytest.fixture
    def rsf_instance(self, mock_model):
        rsf = RSF(
            number_slip_values=5,
            lowest_slip_value=100.0,
            largest_slip_value=5000.0,
            qstart=1000.0,
            qpriors=["Uniform", 0.0, 10000.0],
            plotfigs=False
        )
        rsf.model = mock_model
        return rsf

    def test_generate_time_series(self, rsf_instance):
        # Example test for the generate_time_series method
        result = rsf_instance.generate_time_series()
        assert isinstance(result, np.ndarray), "generate_time_series should return a numpy array"
        assert len(result) == rsf_instance.model.num_tsteps * rsf_instance.num_dc, "Length of result array should match expected size"
