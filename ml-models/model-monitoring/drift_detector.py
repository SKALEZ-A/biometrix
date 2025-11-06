import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_data: np.ndarray, feature_names: List[str]):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.reference_stats = self._calculate_statistics(reference_data)
        
    def _calculate_statistics(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        stats_dict = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_data = data[:, i]
            stats_dict[feature_name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'median': np.median(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75)
            }
        
        return stats_dict
        
    def detect_drift(self, current_data: np.ndarray, threshold: float = 0.05) -> Dict[str, any]:
        """Detect data drift using statistical tests"""
        drift_results = {
            'has_drift': False,
            'drifted_features': [],
            'drift_scores': {},
            'statistical_tests': {}
        }
        
        for i, feature_name in enumerate(self.feature_names):
            reference_feature = self.reference_data[:, i]
            current_feature = current_data[:, i]
            
            # Kolmogorov-Smirnov test
            ks_statistic, ks_pvalue = stats.ks_2samp(reference_feature, current_feature)
            
            # Chi-square test for categorical features
            chi2_statistic, chi2_pvalue = self._chi_square_test(reference_feature, current_feature)
            
            # Population Stability Index
            psi_score = self._calculate_psi(reference_feature, current_feature)
            
            drift_results['statistical_tests'][feature_name] = {
                'ks_statistic': float(ks_statistic),
                'ks_pvalue': float(ks_pvalue),
                'chi2_statistic': float(chi2_statistic),
                'chi2_pvalue': float(chi2_pvalue),
                'psi_score': float(psi_score)
            }
            
            drift_results['drift_scores'][feature_name] = float(psi_score)
            
            if ks_pvalue < threshold or psi_score > 0.2:
                drift_results['has_drift'] = True
                drift_results['drifted_features'].append(feature_name)
        
        logger.info(f"Drift detection completed. Drift detected: {drift_results['has_drift']}")
        return drift_results
        
    def _chi_square_test(self, reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Perform chi-square test"""
        try:
            bins = np.histogram_bin_edges(np.concatenate([reference, current]), bins=10)
            ref_hist, _ = np.histogram(reference, bins=bins)
            cur_hist, _ = np.histogram(current, bins=bins)
            
            # Avoid division by zero
            ref_hist = ref_hist + 1
            cur_hist = cur_hist + 1
            
            chi2_stat = np.sum((ref_hist - cur_hist) ** 2 / ref_hist)
            p_value = 1 - stats.chi2.cdf(chi2_stat, len(bins) - 1)
            
            return chi2_stat, p_value
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return 0.0, 1.0
        
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            bin_edges = np.histogram_bin_edges(reference, bins=bins)
            
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            cur_hist, _ = np.histogram(current, bins=bin_edges)
            
            ref_pct = ref_hist / len(reference)
            cur_pct = cur_hist / len(current)
            
            # Avoid log(0)
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
            
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            return psi
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
