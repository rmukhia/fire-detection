import unittest
import pytest
import time
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from aad.common.config import Config
from aad.common.performance import (
    get_optimal_device,
    get_optimal_worker_count,
    get_memory_info,
    check_system_requirements
)


class TestPerformanceOptimizations(unittest.TestCase):
    
    def test_device_detection(self):
        """Test automatic device detection."""
        device = get_optimal_device()
        self.assertIn(device, ['cpu', 'cuda', 'mps'] + [f'cuda:{i}' for i in range(8)])
    
    def test_worker_count_optimization(self):
        """Test worker count calculation."""
        cpu_workers = get_optimal_worker_count("cpu_bound")
        io_workers = get_optimal_worker_count("io_bound")
        mixed_workers = get_optimal_worker_count("mixed")
        
        self.assertGreaterEqual(cpu_workers, 1)
        self.assertGreaterEqual(io_workers, 1)
        self.assertGreaterEqual(mixed_workers, 1)
        
        # IO-bound should typically allow more workers
        self.assertGreaterEqual(io_workers, cpu_workers)
    
    def test_memory_info(self):
        """Test memory information gathering."""
        info = get_memory_info()
        
        self.assertIn('total_ram_gb', info)
        self.assertIn('available_ram_gb', info)
        self.assertIn('ram_usage_percent', info)
        
        self.assertGreater(info['total_ram_gb'], 0)
        self.assertGreater(info['available_ram_gb'], 0)
    
    def test_system_requirements(self):
        """Test system requirements checking."""
        requirements = check_system_requirements()
        
        self.assertIn('memory_sufficient', requirements)
        self.assertIn('python_version_ok', requirements)
        self.assertIn('torch_available', requirements)
        
        # These should be true in our test environment
        self.assertTrue(requirements['python_version_ok'])
        self.assertTrue(requirements['torch_available'])
    
    def test_config_optimization(self):
        """Test that config uses optimized defaults."""
        config = Config()
        
        # Device should be auto-detected
        self.assertIn(config.training.DEVICE, ['cpu', 'cuda', 'mps'])
        
        # Workers should be reasonable
        self.assertGreaterEqual(config.data_pipeline.NUM_WORKERS, 1)
        self.assertLessEqual(config.data_pipeline.NUM_WORKERS, 32)  # Reasonable upper bound
    
    @pytest.mark.benchmark
    def test_data_loading_benchmark(self, benchmark):
        """Benchmark data loading performance."""
        def load_test_data():
            # Simulate loading sensor data
            data = pd.DataFrame({
                'Datetime': pd.date_range('2023-01-01', periods=1000, freq='2min'),
                'Sensor_Id': np.random.randint(1, 10, 1000),
                'PM2.5': np.random.normal(50, 20, 1000),
                'Carbon dioxide (CO2)': np.random.normal(400, 50, 1000),
                'Relative humidity': np.random.normal(60, 15, 1000)
            })
            return data.groupby('Sensor_Id').agg({
                'PM2.5': 'mean',
                'Carbon dioxide (CO2)': 'mean', 
                'Relative humidity': 'mean'
            })
        
        result = benchmark(load_test_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    @pytest.mark.benchmark  
    def test_tensor_operations_benchmark(self, benchmark):
        """Benchmark tensor operations on available device."""
        device = get_optimal_device()
        
        def tensor_ops():
            x = torch.randn(1000, 100, device=device)
            y = torch.randn(100, 50, device=device)
            z = torch.mm(x, y)
            return torch.sum(z)
        
        result = benchmark(tensor_ops)
        self.assertIsInstance(result, torch.Tensor)
    
    def test_memory_usage_tracking(self):
        """Test memory usage during operations."""
        initial_memory = get_memory_info()
        
        # Allocate some memory
        large_array = np.random.rand(1000000)
        large_tensor = torch.randn(10000, 1000)
        
        current_memory = get_memory_info()
        
        # Memory usage should have increased
        self.assertLess(
            current_memory['available_ram_gb'], 
            initial_memory['available_ram_gb']
        )
        
        # Clean up
        del large_array, large_tensor


class TestConfigOptimizations(unittest.TestCase):
    
    def test_auto_device_selection(self):
        """Test automatic device selection in config."""
        # Test with environment override
        with patch.dict('os.environ', {'DEVICE': 'cpu'}):
            config = Config()
            self.assertEqual(config.training.DEVICE, 'cpu')
    
    def test_worker_count_optimization(self):
        """Test optimized worker count calculation."""
        config = Config()
        
        # Should be at least 1
        self.assertGreaterEqual(config.data_pipeline.NUM_WORKERS, 1)
        
        # Should not exceed reasonable bounds
        self.assertLessEqual(config.data_pipeline.NUM_WORKERS, 16)
    
    def test_batch_size_scaling(self):
        """Test that batch size scales with available memory."""
        config = Config()
        
        # Batch size should be reasonable
        self.assertGreaterEqual(config.training.BATCH_SIZE, 4)
        self.assertLessEqual(config.training.BATCH_SIZE, 256)


if __name__ == '__main__':
    # Run with benchmark support if available
    try:
        pytest.main([__file__, '-v', '--benchmark-only'])
    except ImportError:
        unittest.main()