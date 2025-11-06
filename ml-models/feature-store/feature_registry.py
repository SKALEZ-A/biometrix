import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FeatureRegistry:
    def __init__(self):
        self.features = {}
        self.feature_metadata = {}
        self.feature_versions = {}
        
    def register_feature(self, name: str, description: str, 
                        feature_type: str, source: str, 
                        transformation: Optional[str] = None) -> None:
        """Register a new feature in the registry"""
        feature_id = f"{name}_v1"
        
        self.features[feature_id] = {
            'name': name,
            'description': description,
            'type': feature_type,
            'source': source,
            'transformation': transformation,
            'created_at': datetime.now(),
            'version': 1
        }
        
        self.feature_metadata[name] = {
            'current_version': 1,
            'versions': [feature_id]
        }
        
        logger.info(f"Feature registered: {name}")
        
    def get_feature(self, name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve feature definition"""
        if name not in self.feature_metadata:
            raise ValueError(f"Feature {name} not found")
            
        if version is None:
            version = self.feature_metadata[name]['current_version']
            
        feature_id = f"{name}_v{version}"
        
        if feature_id not in self.features:
            raise ValueError(f"Feature version {feature_id} not found")
            
        return self.features[feature_id]
        
    def update_feature(self, name: str, **kwargs) -> None:
        """Create a new version of an existing feature"""
        if name not in self.feature_metadata:
            raise ValueError(f"Feature {name} not found")
            
        current_version = self.feature_metadata[name]['current_version']
        new_version = current_version + 1
        
        old_feature = self.get_feature(name, current_version)
        new_feature = old_feature.copy()
        new_feature.update(kwargs)
        new_feature['version'] = new_version
        new_feature['updated_at'] = datetime.now()
        
        feature_id = f"{name}_v{new_version}"
        self.features[feature_id] = new_feature
        
        self.feature_metadata[name]['current_version'] = new_version
        self.feature_metadata[name]['versions'].append(feature_id)
        
        logger.info(f"Feature updated: {name} to version {new_version}")
        
    def list_features(self) -> List[Dict[str, Any]]:
        """List all registered features"""
        return [
            {
                'name': name,
                'current_version': meta['current_version'],
                'total_versions': len(meta['versions'])
            }
            for name, meta in self.feature_metadata.items()
        ]
        
    def compute_feature(self, name: str, data: pd.DataFrame) -> pd.Series:
        """Compute feature values from raw data"""
        feature = self.get_feature(name)
        
        if feature['transformation']:
            # Execute transformation logic
            return self._apply_transformation(data, feature['transformation'])
        else:
            # Return raw feature
            return data[feature['source']]
            
    def _apply_transformation(self, data: pd.DataFrame, transformation: str) -> pd.Series:
        """Apply transformation to compute feature"""
        # Simplified transformation execution
        # In production, this would use a proper expression evaluator
        return eval(transformation, {'data': data, 'np': np, 'pd': pd})
        
    def get_feature_statistics(self, name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for a feature"""
        feature_values = self.compute_feature(name, data)
        
        return {
            'mean': float(feature_values.mean()),
            'std': float(feature_values.std()),
            'min': float(feature_values.min()),
            'max': float(feature_values.max()),
            'median': float(feature_values.median()),
            'null_count': int(feature_values.isnull().sum()),
            'unique_count': int(feature_values.nunique())
        }
