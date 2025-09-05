"""
Data validation and quality assurance utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class DataValidator:
    """Comprehensive data validation for fire detection system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_sensor_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate sensor data quality and completeness.
        
        Args:
            df: Sensor data DataFrame
            
        Returns:
            dict: Validation results
        """
        results = {
            'total_records': len(df),
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = ['Datetime', 'Sensor_Id', 'PM2.5', 'Carbon dioxide (CO2)', 'Relative humidity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'Datetime' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Datetime']):
                results['issues'].append("Datetime column is not datetime type")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            results['warnings'].append(f"Missing values found: {missing_counts.to_dict()}")
        
        # Check data ranges
        if 'PM2.5' in df.columns:
            pm25_stats = self._validate_pm25(df['PM2.5'])
            results['stats']['PM2.5'] = pm25_stats
            
        if 'Carbon dioxide (CO2)' in df.columns:
            co2_stats = self._validate_co2(df['Carbon dioxide (CO2)'])
            results['stats']['CO2'] = co2_stats
            
        if 'Relative humidity' in df.columns:
            humidity_stats = self._validate_humidity(df['Relative humidity'])
            results['stats']['humidity'] = humidity_stats
        
        # Check time series continuity
        if 'Datetime' in df.columns and 'Sensor_Id' in df.columns:
            continuity_results = self._check_time_continuity(df)
            results['stats']['continuity'] = continuity_results
        
        self.validation_results['sensor_data'] = results
        return results
    
    def validate_location_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate location data quality.
        
        Args:
            df: Location data DataFrame
            
        Returns:
            dict: Validation results
        """
        results = {
            'total_records': len(df),
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = ['Sensor_Id', 'GPS_Lat', 'GPS_Lon']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check coordinate ranges
        if 'GPS_Lat' in df.columns:
            lat_issues = self._validate_latitude(df['GPS_Lat'])
            if lat_issues:
                results['issues'].extend(lat_issues)
                
        if 'GPS_Lon' in df.columns:
            lon_issues = self._validate_longitude(df['GPS_Lon'])
            if lon_issues:
                results['issues'].extend(lon_issues)
        
        # Check for duplicate sensor IDs
        if 'Sensor_Id' in df.columns:
            duplicates = df['Sensor_Id'].duplicated().sum()
            if duplicates > 0:
                results['warnings'].append(f"Duplicate sensor IDs: {duplicates}")
        
        self.validation_results['location_data'] = results
        return results
    
    def validate_label_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate fire event label data.
        
        Args:
            df: Label data DataFrame
            
        Returns:
            dict: Validation results
        """
        results = {
            'total_records': len(df),
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = ['start_time', 'end_time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Check time logic
        if 'start_time' in df.columns and 'end_time' in df.columns:
            invalid_times = df['start_time'] >= df['end_time']
            if invalid_times.sum() > 0:
                results['issues'].append(f"Invalid time ranges: {invalid_times.sum()} records")
        
        # Check geometry if present
        if 'geometry' in df.columns:
            geometry_issues = self._validate_geometry(df['geometry'])
            if geometry_issues:
                results['warnings'].extend(geometry_issues)
        
        self.validation_results['label_data'] = results
        return results
    
    def _validate_pm25(self, series: pd.Series) -> Dict[str, Any]:
        """Validate PM2.5 measurements."""
        stats = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std(),
            'outliers': 0,
            'negative_values': 0
        }
        
        # Check for negative values
        stats['negative_values'] = (series < 0).sum()
        
        # Check for extreme values (likely outliers)
        q99 = series.quantile(0.99)
        stats['outliers'] = (series > q99 * 3).sum()
        
        return stats
    
    def _validate_co2(self, series: pd.Series) -> Dict[str, Any]:
        """Validate CO2 measurements."""
        stats = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std(),
            'outliers': 0,
            'below_atmospheric': 0
        }
        
        # Check for values below atmospheric CO2 (~400 ppm)
        stats['below_atmospheric'] = (series < 300).sum()
        
        # Check for extreme values
        stats['outliers'] = (series > 2000).sum()  # Very high CO2 levels
        
        return stats
    
    def _validate_humidity(self, series: pd.Series) -> Dict[str, Any]:
        """Validate humidity measurements."""
        stats = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std(),
            'out_of_range': 0
        }
        
        # Humidity should be 0-100%
        stats['out_of_range'] = ((series < 0) | (series > 100)).sum()
        
        return stats
    
    def _validate_latitude(self, series: pd.Series) -> List[str]:
        """Validate latitude values."""
        issues = []
        
        # Latitude should be -90 to 90
        out_of_range = ((series < -90) | (series > 90)).sum()
        if out_of_range > 0:
            issues.append(f"Latitude out of range: {out_of_range} values")
        
        return issues
    
    def _validate_longitude(self, series: pd.Series) -> List[str]:
        """Validate longitude values."""
        issues = []
        
        # Longitude should be -180 to 180
        out_of_range = ((series < -180) | (series > 180)).sum()
        if out_of_range > 0:
            issues.append(f"Longitude out of range: {out_of_range} values")
        
        return issues
    
    def _validate_geometry(self, series: pd.Series) -> List[str]:
        """Validate geometry strings."""
        warnings = []
        
        # Check for null geometries
        null_geom = series.isnull().sum()
        if null_geom > 0:
            warnings.append(f"Null geometries: {null_geom}")
        
        # Basic format check for WKT
        if not series.isnull().all():
            invalid_format = 0
            for geom in series.dropna():
                if not isinstance(geom, str) or not geom.startswith(('POINT', 'POLYGON', 'MULTIPOLYGON')):
                    invalid_format += 1
            
            if invalid_format > 0:
                warnings.append(f"Invalid geometry format: {invalid_format}")
        
        return warnings
    
    def _check_time_continuity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check time series continuity for each sensor."""
        continuity_stats = {}
        
        for sensor_id in df['Sensor_Id'].unique():
            sensor_data = df[df['Sensor_Id'] == sensor_id].sort_values('Datetime')
            
            # Calculate time gaps
            time_diffs = sensor_data['Datetime'].diff()
            
            stats = {
                'total_records': len(sensor_data),
                'time_span_hours': (sensor_data['Datetime'].max() - sensor_data['Datetime'].min()).total_seconds() / 3600,
                'mean_interval_minutes': time_diffs.mean().total_seconds() / 60 if len(time_diffs) > 1 else None,
                'max_gap_hours': time_diffs.max().total_seconds() / 3600 if len(time_diffs) > 1 else None,
                'gaps_over_1h': (time_diffs > pd.Timedelta('1h')).sum() if len(time_diffs) > 1 else 0
            }
            
            continuity_stats[sensor_id] = stats
        
        return continuity_stats
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        report = ["=" * 50]
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 50)
        
        for data_type, results in self.validation_results.items():
            report.append(f"\n{data_type.upper()}:")
            report.append(f"  Total records: {results['total_records']}")
            
            if results['issues']:
                report.append("  ISSUES:")
                for issue in results['issues']:
                    report.append(f"    ❌ {issue}")
            
            if results['warnings']:
                report.append("  WARNINGS:")
                for warning in results['warnings']:
                    report.append(f"    ⚠️  {warning}")
            
            if not results['issues'] and not results['warnings']:
                report.append("    ✅ No issues found")
        
        return "\n".join(report)
    
    def save_report(self, filepath: Path):
        """Save validation report to file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Validation report saved to {filepath}")


def validate_data_pipeline(config, logger: Optional[logging.Logger] = None) -> DataValidator:
    """
    Run complete data validation for the pipeline.
    
    Args:
        config: Configuration object
        logger: Optional logger
        
    Returns:
        DataValidator: Validator with results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    validator = DataValidator(logger)
    
    # Validate sensor data if it exists
    sensor_path = Path(config.paths.RAW_DATA_PATH)
    if sensor_path.exists():
        logger.info(f"Validating sensor data: {sensor_path}")
        try:
            df_sensor = pd.read_csv(sensor_path, parse_dates=['Datetime'])
            validator.validate_sensor_data(df_sensor)
        except Exception as e:
            logger.error(f"Error validating sensor data: {e}")
    
    # Validate location data if it exists
    location_path = Path(config.paths.LOCATION_DATA_PATH)
    if location_path.exists():
        logger.info(f"Validating location data: {location_path}")
        try:
            df_location = pd.read_csv(location_path)
            validator.validate_location_data(df_location)
        except Exception as e:
            logger.error(f"Error validating location data: {e}")
    
    # Validate label data if it exists
    label_path = Path(config.paths.LABEL_DATA_PATH)
    if label_path.exists():
        logger.info(f"Validating label data: {label_path}")
        try:
            df_labels = pd.read_csv(label_path, parse_dates=['start_time', 'end_time'])
            validator.validate_label_data(df_labels)
        except Exception as e:
            logger.error(f"Error validating label data: {e}")
    
    # Log summary
    logger.info("\n" + validator.generate_report())
    
    return validator


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from aad.common.config import Config
    from aad.common.core_logging import ProcessLogger
    
    config = Config()
    logger = ProcessLogger(config, 'data_validation')
    
    validator = validate_data_pipeline(config, logger)
    
    # Save report
    report_path = Path(config.paths.OUTPUT_DIR) / "data_validation_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    validator.save_report(report_path)