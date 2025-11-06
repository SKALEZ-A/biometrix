import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import json
import logging
from enum import Enum
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure logging for XAI operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations supported by the XAI system."""
    SHAP = "shap"  # SHAP (SHapley Additive exPlanations)
    LIME = "lime"  # Local Interpretable Model-agnostic Explanations
    COUNTERFACTUAL = "counterfactual"  # Counterfactual explanations
    FEATURE_IMPORTANCE = "feature_importance"  # Global feature importance
    DECISION_TREE = "decision_tree"  # Decision tree visualization
    PDP = "pdp"  # Partial Dependence Plots
    ICE = "ice"  # Individual Conditional Expectation plots

@dataclass
class FraudExplanation:
    """Data class representing a complete fraud explanation."""
    prediction: float  # Raw prediction score (0-1)
    prediction_label: str  # 'fraud' or 'legitimate'
    confidence: float  # Model confidence [0, 1]
    explanation_type: ExplanationType
    feature_importance: Dict[str, float]  # Feature contributions
    global_importance: Optional[Dict[str, float]] = None  # Global feature importance
    counterfactuals: Optional[List[Dict[str, Any]]] = None  # What-if scenarios
    visual_explanation: Optional[str] = None  # Base64 encoded visualization
    regulatory_compliance: Dict[str, str] = None  # Compliance annotations
    timestamp: float = None
    model_version: str = "v1.0"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class ModelMetadata:
    """Stores metadata about the fraud detection model for explainability."""
    
    def __init__(self, model_name: str, model_type: str, features: List[str], 
                 target_name: str = "fraud_probability", version: str = "1.0"):
        self.model_name = model_name
        self.model_type = model_type
        self.features = features
        self.target_name = target_name
        self.version = version
        self.feature_descriptions: Dict[str, str] = {}
        self.business_impact: Dict[str, float] = {}  # Sensitivity weights
        self.compliance_requirements: Dict[str, List[str]] = {}
        
        # Default business impact weights for fraud detection
        self._set_default_business_impact()
    
    def _set_default_business_impact(self):
        """Set default business impact weights for common fraud features."""
        impact_mapping = {
            'transaction_amount': 0.25,
            'account_age_days': 0.15,
            'velocity_count': 0.20,
            'geolocation_distance': 0.18,
            'device_fingerprint_score': 0.12,
            'behavioral_anomaly': 0.10,
            'ip_reputation': 0.08,
            'time_of_day': 0.05,
            'email_domain_age': 0.04,
            'merchant_category': 0.03
        }
        
        for feature, impact in impact_mapping.items():
            if feature in self.features:
                self.business_impact[feature] = impact
    
    def add_feature_description(self, feature: str, description: str, 
                              business_critical: bool = False,
                              compliance_tags: List[str] = None):
        """Add a human-readable description for a feature."""
        self.feature_descriptions[feature] = {
            'description': description,
            'business_critical': business_critical,
            'compliance_tags': compliance_tags or [],
            'example_range': None  # Can be set later
        }
        
        if business_critical and feature not in self.business_impact:
            self.business_impact[feature] = 0.15  # Default high impact
    
    def set_feature_range(self, feature: str, min_val: float, max_val: float, 
                         unit: str = ""):
        """Set the valid range for a feature for normalization."""
        if feature in self.feature_descriptions:
            self.feature_descriptions[feature]['example_range'] = {
                'min': min_val,
                'max': max_val,
                'unit': unit
            }

class RegulatoryCompliance:
    """Handles regulatory compliance requirements for fraud explanations."""
    
    def __init__(self, jurisdiction: str = "US", regulations: List[str] = None):
        self.jurisdiction = jurisdiction
        self.regulations = regulations or ["GDPR", "CCPA", "FCRA", "GLBA"]
        self.required_explanations: Dict[str, List[ExplanationType]] = {
            "GDPR": [ExplanationType.COUNTERFACTUAL, ExplanationType.FEATURE_IMPORTANCE],
            "CCPA": [ExplanationType.FEATURE_IMPORTANCE, ExplanationType.PDP],
            "FCRA": [ExplanationType.COUNTERFACTUAL, ExplanationType.DECISION_TREE],
            "GLBA": [ExplanationType.FEATURE_IMPORTANCE, ExplanationType.PDP]
        }
        self.audit_log: List[Dict[str, Any]] = []
    
    def validate_explanation(self, explanation: FraudExplanation) -> Dict[str, Any]:
        """Validate that an explanation meets regulatory requirements."""
        compliance_status = {}
        missing_requirements = []
        
        for regulation in self.regulations:
            required_types = self.required_explanations.get(regulation, [])
            met_requirements = []
            
            if explanation.explanation_type in required_types:
                met_requirements.append(explanation.explanation_type.value)
            
            if explanation.global_importance and ExplanationType.FEATURE_IMPORTANCE in required_types:
                met_requirements.append("feature_importance")
            
            if explanation.counterfactuals and ExplanationType.COUNTERFACTUAL in required_types:
                met_requirements.append("counterfactual")
            
            compliance_status[regulation] = {
                'met': len(met_requirements) >= len(required_types),
                'required': [t.value for t in required_types],
                'provided': met_requirements,
                'complete': len(met_requirements) == len(required_types)
            }
            
            if not compliance_status[regulation]['complete']:
                missing_requirements.extend([t for t in required_types if t.value not in met_requirements])
        
        # Log compliance validation
        self.audit_log.append({
            'timestamp': time.time(),
            'explanation_id': f"exp_{int(time.time())}",
            'compliance_status': compliance_status,
            'missing_requirements': list(set(missing_requirements))
        })
        
        return {
            'compliant': len(missing_requirements) == 0,
            'details': compliance_status,
            'recommendations': self._generate_compliance_recommendations(missing_requirements)
        }
    
    def _generate_compliance_recommendations(self, missing_types: List[ExplanationType]) -> List[str]:
        """Generate recommendations for missing compliance requirements."""
        recommendations = []
        
        for exp_type in missing_types:
            if exp_type == ExplanationType.COUNTERFACTUAL:
                recommendations.append("Generate counterfactual explanations showing minimal changes needed for different decision")
            elif exp_type == ExplanationType.FEATURE_IMPORTANCE:
                recommendations.append("Include global feature importance rankings with business impact scores")
            elif exp_type == ExplanationType.PDP:
                recommendations.append("Add Partial Dependence Plots for top 3 most influential features")
            elif exp_type == ExplanationType.DECISION_TREE:
                recommendations.append("Provide simplified decision tree visualization for key decision paths")
        
        return recommendations

class SHAPExplainer:
    """SHAP-based explainability for fraud detection models."""
    
    def __init__(self, model, model_metadata: ModelMetadata, background_data: np.ndarray):
        self.model = model
        self.metadata = model_metadata
        self.explainer = None
        self.background_data = background_data
        self.feature_names = model_metadata.features
        
        # Initialize SHAP explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        try:
            if hasattr(self.model, 'predict_proba'):
                # Tree-based models (Random Forest, XGBoost, etc.)
                if 'RandomForest' in str(type(self.model)) or 'XGB' in str(type(self.model)):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Kernel explainer for other models
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        self.background_data
                    )
            else:
                # For regression or custom models
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    self.background_data
                )
            
            logger.info(f"SHAP explainer initialized for {self.metadata.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            # Fallback to basic kernel explainer
            self.explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x)[:, 1] if hasattr(self.model, 'predict_proba') 
                else self.model.predict(x),
                self.background_data[:100]  # Use smaller sample for speed
            )
    
    def explain_instance(self, instance: np.ndarray, nsamples: int = 1000) -> Dict[str, Any]:
        """Generate SHAP explanation for a single instance."""
        try:
            # Ensure instance is properly shaped
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(instance, nsamples=nsamples)
            
            # Handle multi-class case (keep only fraud class)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Fraud class (index 1)
            elif len(instance.shape) == 2 and instance.shape[0] == 1:
                shap_values = shap_values[0]
            
            # Get prediction
            prediction = self.model.predict_proba(instance)[:, 1][0] if hasattr(self.model, 'predict_proba') else self.model.predict(instance)[0]
            
            # Create feature importance mapping
            feature_contributions = {}
            for i, feature in enumerate(self.feature_names):
                contribution = shap_values[i] if len(shap_values.shape) == 1 else shap_values[0, i]
                feature_contributions[feature] = float(contribution)
            
            # Calculate business impact adjusted importance
            business_importance = self._calculate_business_importance(feature_contributions)
            
            explanation_data = {
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                'prediction': float(prediction),
                'feature_contributions': feature_contributions,
                'business_importance': business_importance,
                'expected_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0,
                'top_features': self._get_top_features(feature_contributions, n=5),
                'risk_drivers': self._identify_risk_drivers(feature_contributions),
                'visualization_data': self._prepare_visualization_data(shap_values, instance[0])
            }
            
            logger.info(f"SHAP explanation generated for instance with prediction {prediction:.3f}")
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return self._generate_fallback_explanation(instance)
    
    def _calculate_business_importance(self, contributions: Dict[str, float]) -> Dict[str, float]:
        """Calculate business impact adjusted feature importance."""
        business_scores = {}
        
        for feature, shap_value in contributions.items():
            business_weight = self.metadata.business_impact.get(feature, 0.1)
            business_scores[feature] = abs(shap_value) * business_weight
        
        return dict(sorted(business_scores.items(), key=lambda x: x[1], reverse=True))
    
    def _get_top_features(self, contributions: Dict[str, float], n: int = 5) -> List[Dict[str, Any]]:
        """Get top N most influential features."""
        sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = []
        
        for feature, value in sorted_features[:n]:
            desc = self.metadata.feature_descriptions.get(feature, {}).get('description', feature)
            top_features.append({
                'feature': feature,
                'shap_value': float(value),
                'importance': abs(value),
                'description': desc,
                'business_critical': self.metadata.feature_descriptions.get(feature, {}).get('business_critical', False),
                'direction': 'increases_risk' if value > 0 else 'decreases_risk'
            })
        
        return top_features
    
    def _identify_risk_drivers(self, contributions: Dict[str, float]) -> List[str]:
        """Identify key risk drivers based on business rules."""
        risk_drivers = []
        high_impact_threshold = 0.1
        
        for feature, value in contributions.items():
            if abs(value) > high_impact_threshold:
                if value > 0:  # Increases fraud risk
                    risk_drivers.append(f"{feature} significantly increases fraud risk")
                else:
                    risk_drivers.append(f"{feature} mitigates fraud risk")
        
        return risk_drivers
    
    def _prepare_visualization_data(self, shap_values: np.ndarray, instance: np.ndarray) -> Dict[str, Any]:
        """Prepare data for SHAP visualizations."""
        try:
            # Force plot data (simplified for JSON serialization)
            force_data = {
                'features': self.feature_names,
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else [float(shap_values)],
                'instance_values': instance.tolist(),
                'expected_value': float(self.explainer.expected_value)
            }
            
            # Waterfall plot data
            sorted_idx = np.argsort(shap_values)[::-1]
            waterfall_data = {
                'feature_order': [self.feature_names[i] for i in sorted_idx],
                'shap_values': [float(shap_values[i]) for i in sorted_idx],
                'feature_values': [float(instance[i]) for i in sorted_idx]
            }
            
            return {
                'force_plot': force_data,
                'waterfall_plot': waterfall_data,
                'summary_plot_features': self._prepare_summary_plot_data()
            }
            
        except Exception as e:
            logger.warning(f"Could not prepare visualization data: {e}")
            return {}
    
    def _prepare_summary_plot_data(self) -> List[Dict[str, Any]]:
        """Prepare data for SHAP summary plots."""
        # This would typically require the full dataset and multiple explanations
        # For now, return mock data structure
        return [
            {
                'feature': feature,
                'mean_shap_value': 0.05,  # Would be calculated from multiple instances
                'shap_range': [-0.1, 0.2],
                'feature_range': [0, 1]
            }
            for feature in self.feature_names[:5]  # Top 5 features
        ]
    
    def _generate_fallback_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """Generate a fallback explanation when SHAP fails."""
        prediction = self.model.predict_proba(instance)[:, 1][0] if hasattr(self.model, 'predict_proba') else self.model.predict(instance)[0]
        
        # Simple feature importance based on model coefficients or random forest feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.random.rand(len(self.feature_names))  # Last resort
        
        feature_contributions = {
            feature: float(importances[i] * prediction) 
            for i, feature in enumerate(self.feature_names)
        }
        
        return {
            'shap_values': [],
            'prediction': float(prediction),
            'feature_contributions': feature_contributions,
            'business_importance': {k: v * 0.1 for k, v in feature_contributions.items()},
            'expected_value': 0.5,
            'top_features': self._get_top_features(feature_contributions),
            'risk_drivers': ['Fallback explanation - limited model interpretability'],
            'visualization_data': {}
        }

class LIMEExplainer:
    """LIME-based explainability for local model interpretation."""
    
    def __init__(self, model, model_metadata: ModelMetadata, training_data: pd.DataFrame):
        self.model = model
        self.metadata = model_metadata
        self.feature_names = model_metadata.features
        self.training_data = training_data
        self.scaler = StandardScaler()
        
        # Prepare data for LIME
        self._prepare_lime_data()
    
    def _prepare_lime_data(self):
        """Prepare training data for LIME explanations."""
        try:
            # Select relevant features
            feature_data = self.training_data[self.feature_names].values
            
            # Scale features for better perturbation sampling
            if len(feature_data) > 0:
                self.scaler.fit(feature_data)
                self.scaled_feature_means = self.scaler.transform(feature_data).mean(axis=0)
            else:
                self.scaled_feature_means = np.zeros(len(self.feature_names))
                
        except Exception as e:
            logger.error(f"Error preparing LIME data: {e}")
            self.scaled_feature_means = np.zeros(len(self.feature_names))
    
    def explain_instance(self, instance: Dict[str, Any], num_samples: int = 5000, 
                        num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanation for a single instance."""
        try:
            # Convert instance to numpy array
            instance_array = np.array([[
                instance.get(feature, 0) for feature in self.feature_names
            ]])
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.training_data[self.feature_names].values,
                feature_names=self.feature_names,
                class_names=['legitimate', 'fraud'],
                mode='classification',
                discretize_continuous=True,
                sample_around_instance=True
            )
            
            # Generate explanation
            explanation = explainer.explain_instance(
                data_row=instance_array[0],
                predict_fn=lambda x: self.model.predict_proba(x)[:, 1],
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract LIME results
            lime_values = explanation.as_list()
            
            # Create feature importance mapping
            feature_importance = {feature: 0.0 for feature in self.feature_names}
            for feature, value in lime_values:
                feature_importance[feature] = abs(value)
            
            # Get prediction
            prediction = self.model.predict_proba(instance_array)[:, 1][0]
            
            # Prepare explanation data
            explanation_data = {
                'lime_segments': lime_values,
                'feature_importance': feature_importance,
                'prediction': float(prediction),
                'coverage': explanation.coverage(),
                'local_fidelity': explanation.score,
                'top_features': self._format_top_lime_features(lime_values, num_features=5),
                'interpretable_model': {
                    'coefficients': explanation.interpretable_model_.coef_[0].tolist(),
                    'intercept': float(explanation.interpretable_model_.intercept_[0])
                },
                'visualization_data': self._prepare_lime_visualization(explanation)
            }
            
            logger.info(f"LIME explanation generated with {len(lime_values)} segments")
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return self._generate_lime_fallback(instance)
    
    def _format_top_lime_features(self, lime_segments: List[Tuple[str, float]], 
                                num_features: int = 5) -> List[Dict[str, Any]]:
        """Format top LIME features for display."""
        top_features = []
        sorted_segments = sorted(lime_segments, key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature, value) in enumerate(sorted_segments[:num_features]):
            description = self.metadata.feature_descriptions.get(feature, {}).get('description', feature)
            top_features.append({
                'rank': i + 1,
                'feature': feature,
                'lime_value': float(value),
                'importance': abs(value),
                'description': description,
                'contribution': 'positive' if value > 0 else 'negative',
                'approximate': True  # LIME provides local approximations
            })
        
        return top_features
    
    def _prepare_lime_visualization(self, explanation) -> Dict[str, Any]:
        """Prepare visualization data for LIME explanation."""
        try:
            # Bar chart data for feature contributions
            features = [seg[0] for seg in explanation.as_list()]
            values = [seg[1] for seg in explanation.as_list()]
            
            viz_data = {
                'bar_chart': {
                    'features': features,
                    'values': [float(v) for v in values],
                    'colors': ['red' if v > 0 else 'blue' for v in values]
                },
                'local_model_equation': self._format_lime_equation(explanation),
                'perturbation_samples': len(explanation.domain_mapper.discretizer.kmeans.cluster_centers_)
            }
            
            return viz_data
            
        except Exception as e:
            logger.warning(f"Could not prepare LIME visualization: {e}")
            return {}
    
    def _format_lime_equation(self, explanation) -> str:
        """Format the interpretable LIME model as a human-readable equation."""
        try:
            coefs = explanation.interpretable_model_.coef_[0]
            intercept = explanation.interpretable_model_.intercept_[0]
            
            terms = []
            for i, coef in enumerate(coefs):
                if abs(coef) > 0.001:  # Significant coefficients only
                    feature = self.feature_names[i]
                    sign = '+' if coef > 0 else '-'
                    terms.append(f"{sign} {abs(coef):.3f}*{feature}")
            
            equation = f"f(x) = {intercept:.3f} {' '.join(terms)}"
            return equation
            
        except Exception:
            return "f(x) = interpretable linear model (coefficients available in raw data)"
    
    def _generate_lime_fallback(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback LIME explanation."""
        instance_array = np.array([[
            instance.get(feature, 0) for feature in self.feature_names
        ]])
        
        prediction = self.model.predict_proba(instance_array)[:, 1][0]
        
        # Simple linear approximation as fallback
        feature_importance = {
            feature: abs(np.random.normal(0, 0.05)) * prediction 
            for feature in self.feature_names[:5]  # Top 5 features
        }
        
        return {
            'lime_segments': [],
            'feature_importance': feature_importance,
            'prediction': float(prediction),
            'coverage': 0.0,
            'local_fidelity': 0.0,
            'top_features': [],
            'interpretable_model': {'coefficients': [], 'intercept': 0.5},
            'visualization_data': {}
        }

class CounterfactualExplainer:
    """Generates counterfactual explanations for fraud decisions."""
    
    def __init__(self, model, model_metadata: ModelMetadata, 
                 feature_constraints: Optional[Dict[str, Tuple[float, float]]] = None):
        self.model = model
        self.metadata = model_metadata
        self.feature_names = model_metadata.features
        self.feature_constraints = feature_constraints or {}
        
        # Set default constraints for common fraud features
        self._set_default_constraints()
    
    def _set_default_constraints(self):
        """Set reasonable constraints for fraud detection features."""
        default_constraints = {
            'transaction_amount': (0, 1000000),
            'account_age_days': (0, 365*10),
            'velocity_count': (0, 100),
            'geolocation_distance': (0, 20000),  # km
            'behavioral_anomaly': (0, 1),
            'ip_reputation': (0, 1),
            'device_fingerprint_score': (0, 1),
            'time_of_day': (0, 24)
        }
        
        for feature, (min_val, max_val) in default_constraints.items():
            if feature in self.feature_names and feature not in self.feature_constraints:
                self.feature_constraints[feature] = (min_val, max_val)
    
    def generate_counterfactuals(self, instance: Dict[str, Any], target_class: str = 'legitimate',
                               max_changes: int = 3, total_distance_threshold: float = 0.2,
                               business_priority: List[str] = None) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations showing how to change the decision."""
        try:
            # Convert instance to array
            instance_array = np.array([[
                instance.get(feature, 0) for feature in self.feature_names
            ]])
            
            current_prediction = self.model.predict_proba(instance_array)[:, 1][0]
            current_label = 'fraud' if current_prediction > 0.5 else 'legitimate'
            
            if current_label == target_class:
                logger.info(f"Instance already has target class {target_class}")
                return []
            
            # Priority features for changes (business critical first)
            priority_features = business_priority or [
                f for f in self.metadata.business_impact.keys() 
                if self.metadata.business_impact[f] > 0.1
            ]
            
            # Generate counterfactuals by systematically changing features
            counterfactuals = []
            feature_importance = self._get_feature_importance()
            
            # Strategy 1: Change most influential features
            for i, feature in enumerate(priority_features[:max_changes]):
                cf_instance = instance.copy()
                
                # Determine direction of change based on feature type
                current_value = instance.get(feature, 0)
                min_val, max_val = self.feature_constraints.get(feature, (0, 1))
                
                if current_prediction > 0.5:  # Currently fraud, reduce risk
                    # For positive impact features, reduce value
                    if feature_importance.get(feature, 0) > 0:
                        new_value = max(min_val, current_value * 0.7)
                    else:
                        new_value = min(max_val, current_value * 1.3)
                else:  # Currently legitimate, increase risk (for completeness)
                    if feature_importance.get(feature, 0) > 0:
                        new_value = min(max_val, current_value * 1.3)
                    else:
                        new_value = max(min_val, current_value * 0.7)
                
                # Apply constraints
                new_value = np.clip(new_value, min_val, max_val)
                cf_instance[feature] = new_value
                
                # Calculate distance (simple Euclidean)
                distance = self._calculate_distance(instance_array[0], 
                                                  np.array([cf_instance[f] for f in self.feature_names]))
                
                if distance <= total_distance_threshold:
                    cf_array = np.array([[
                        cf_instance.get(f, 0) for f in self.feature_names
                    ]])
                    
                    new_prediction = self.model.predict_proba(cf_array)[:, 1][0]
                    new_label = 'fraud' if new_prediction > 0.5 else 'legitimate'
                    
                    counterfactuals.append({
                        'changed_feature': feature,
                        'original_value': instance.get(feature, 0),
                        'new_value': new_value,
                        'change_magnitude': abs(new_value - instance.get(feature, 0)),
                        'new_prediction': float(new_prediction),
                        'new_label': new_label,
                        'distance': float(distance),
                        'description': self._generate_cf_description(feature, new_value, new_label),
                        'business_impact': self.metadata.business_impact.get(feature, 0.1),
                        'compliance_note': self._get_compliance_note(feature)
                    })
            
            # Strategy 2: Combination of 2 features if single changes insufficient
            if len(counterfactuals) < 2:
                counterfactuals.extend(self._generate_combination_counterfactuals(
                    instance, max_changes=2, total_distance_threshold=total_distance_threshold
                ))
            
            # Sort by distance (most similar first)
            counterfactuals.sort(key=lambda x: x['distance'])
            
            logger.info(f"Generated {len(counterfactuals)} counterfactuals for {target_class} target")
            return counterfactuals[:max_changes]  # Return top N
            
        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            return []
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model (SHAP fallback if needed)."""
        if hasattr(self.model, 'feature_importances_'):
            return {self.feature_names[i]: float(self.model.feature_importances_[i]) 
                   for i in range(len(self.feature_names))}
        elif hasattr(self.model, 'coef_'):
            return {self.feature_names[i]: float(abs(self.model.coef_[0][i])) 
                   for i in range(len(self.feature_names))}
        else:
            # Fallback: business impact weights
            return self.metadata.business_impact
    
    def _calculate_distance(self, original: np.ndarray, counterfactual: np.ndarray) -> float:
        """Calculate normalized Euclidean distance between instances."""
        # Normalize features before calculating distance
        differences = (original - counterfactual) ** 2
        distance = np.sqrt(np.sum(differences))
        
        # Normalize by feature ranges
        normalized_distance = 0
        for i, feature in enumerate(self.feature_names):
            min_val, max_val = self.feature_constraints.get(feature, (0, 1))
            range_val = max_val - min_val if max_val > min_val else 1
            normalized_distance += ((original[i] - counterfactual[i]) / range_val) ** 2
        
        normalized_distance = np.sqrt(normalized_distance) / np.sqrt(len(self.feature_names))
        return min(1.0, float(normalized_distance))
    
    def _generate_combination_counterfactuals(self, instance: Dict[str, Any], 
                                           max_changes: int, total_distance_threshold: float) -> List[Dict[str, Any]]:
        """Generate counterfactuals by changing combinations of features."""
        priority_features = list(self.metadata.business_impact.keys())[:max_changes]
        combinations = []
        
        # Generate all combinations of 2 features
        from itertools import combinations
        for feature_pair in combinations(priority_features, 2):
            cf_instance = instance.copy()
            distance = 0
            changes = []
            
            for feature in feature_pair:
                current_value = instance.get(feature, 0)
                min_val, max_val = self.feature_constraints.get(feature, (0, 1))
                
                # Similar logic to single feature changes
                if 'amount' in feature or 'velocity' in feature:
                    new_value = max(min_val, current_value * 0.6)
                else:
                    new_value = max(min_val, current_value * 0.8)
                
                new_value = np.clip(new_value, min_val, max_val)
                cf_instance[feature] = new_value
                changes.append({
                    'feature': feature,
                    'original': current_value,
                    'new': new_value
                })
                distance += abs(new_value - current_value)
            
            if distance / len(feature_pair) <= total_distance_threshold * 1.5:
                combinations.append({
                    'changed_features': changes,
                    'total_distance': distance / len(feature_pair),
                    'description': f"Change {len(feature_pair)} features to achieve different outcome"
                })
        
        return combinations
    
    def _generate_cf_description(self, feature: str, new_value: float, 
                               new_label: str) -> str:
        """Generate human-readable description for counterfactual."""
        desc = self.metadata.feature_descriptions.get(feature, {}).get('description', feature)
        range_info = self.metadata.feature_descriptions.get(feature, {}).get('example_range', {})
        
        if new_label == 'legitimate':
            return f"Reduce {desc} to {new_value:.2f} {range_info.get('unit', '')} to be approved"
        else:
            return f"Increase {desc} to {new_value:.2f} {range_info.get('unit', '')} to trigger fraud detection"
    
    def _get_compliance_note(self, feature: str) -> str:
        """Get compliance note for the feature change."""
        tags = self.metadata.feature_descriptions.get(feature, {}).get('compliance_tags', [])
        if 'sensitive' in tags:
            return "This change affects sensitive personal data - requires additional verification"
        elif 'pii' in tags:
            return "Personally identifiable information modification - log for audit purposes"
        return "Standard feature modification"

class FraudXAI:
    """Main orchestrator for explainable AI in fraud detection systems."""
    
    def __init__(self, model, model_metadata: ModelMetadata, training_data: pd.DataFrame,
                 background_data: Optional[np.ndarray] = None, jurisdiction: str = "US"):
        self.model = model
        self.metadata = model_metadata
        self.training_data = training_data
        self.background_data = background_data or training_data.sample(min(100, len(training_data))).values
        self.jurisdiction = jurisdiction
        self.compliance_checker = RegulatoryCompliance(jurisdiction)
        
        # Initialize explainers
        self.shap_explainer = SHAPExplainer(model, model_metadata, self.background_data)
        self.lime_explainer = LIMEExplainer(model, model_metadata, training_data)
        self.counterfactual_explainer = CounterfactualExplainer(model, model_metadata)
        
        # Track generated explanations for audit
        self.explanation_history: List[FraudExplanation] = []
        self.performance_metrics = {}
    
    def generate_comprehensive_explanation(self, instance: Dict[str, Any], 
                                        explanation_types: List[ExplanationType] = None,
                                        target_class: str = 'legitimate') -> FraudExplanation:
        """Generate a comprehensive explanation using multiple techniques."""
        if explanation_types is None:
            explanation_types = [ExplanationType.SHAP, ExplanationType.LIME, ExplanationType.COUNTERFACTUAL]
        
        try:
            # Get raw prediction
            instance_array = np.array([[
                instance.get(feature, 0) for feature in self.metadata.features
            ]])
            
            prediction = self.model.predict_proba(instance_array)[:, 1][0]
            prediction_label = 'fraud' if prediction > 0.5 else 'legitimate'
            confidence = max(prediction, 1 - prediction)
            
            # Generate explanations for each requested type
            explanations = {}
            feature_importance = {}
            
            if ExplanationType.SHAP in explanation_types:
                shap_data = self.shap_explainer.explain_instance(instance_array[0])
                explanations['shap'] = shap_data
                feature_importance.update(shap_data['business_importance'])
            
            if ExplanationType.LIME in explanation_types:
                lime_data = self.lime_explainer.explain_instance(instance, num_samples=2000)
                explanations['lime'] = lime_data
                # Combine with SHAP (average importance)
                for f, imp in lime_data['feature_importance'].items():
                    feature_importance[f] = feature_importance.get(f, 0) * 0.7 + imp * 0.3
            
            if ExplanationType.COUNTERFACTUAL in explanation_types:
                cf_data = self.counterfactual_explainer.generate_counterfactuals(
                    instance, target_class=target_class, max_changes=3
                )
                explanations['counterfactual'] = cf_data
            
            if ExplanationType.FEATURE_IMPORTANCE in explanation_types:
                global_importance = self._calculate_global_importance()
                explanations['global_importance'] = global_importance
                feature_importance.update(global_importance)
            
            # Generate visualizations (if possible)
            visual_explanation = self._generate_visual_summary(explanations, instance)
            
            # Compliance validation
            compliance_report = self.compliance_checker.validate_explanation(
                FraudExplanation(
                    prediction=prediction,
                    prediction_label=prediction_label,
                    confidence=confidence,
                    explanation_type=explanation_types[0],  # Primary type
                    feature_importance=feature_importance,
                    global_importance=explanations.get('global_importance', {}),
                    counterfactuals=explanations.get('counterfactual', []),
                    visual_explanation=visual_explanation
                )
            )
            
            # Create complete explanation
            full_explanation = FraudExplanation(
                prediction=prediction,
                prediction_label=prediction_label,
                confidence=confidence,
                explanation_type=explanation_types[0],
                feature_importance=feature_importance,
                global_importance=explanations.get('global_importance', None),
                counterfactuals=explanations.get('counterfactual', None),
                visual_explanation=visual_explanation,
                regulatory_compliance=compliance_report['details'],
                model_version=self.metadata.version
            )
            
            # Store for audit
            self.explanation_history.append(full_explanation)
            
            # Log performance metrics
            self._update_performance_metrics(full_explanation)
            
            logger.info(f"Comprehensive explanation generated for {prediction_label} prediction (confidence: {confidence:.3f})")
            return full_explanation
            
        except Exception as e:
            logger.error(f"Error generating comprehensive explanation: {e}")
            # Generate minimal explanation as fallback
            return self._generate_minimal_explanation(instance, prediction_label, confidence)
    
    def _calculate_global_importance(self) -> Dict[str, float]:
        """Calculate global feature importance using permutation importance or model built-ins."""
        try:
            from sklearn.inspection import permutation_importance
            
            # Use a sample of training data for efficiency
            sample_size = min(1000, len(self.training_data))
            sample_data = self.training_data[self.metadata.features].sample(sample_size).values
            
            # Calculate permutation importance
            result = permutation_importance(
                self.model, 
                sample_data, 
                self.training_data['fraud'].iloc[:sample_size].values,
                n_repeats=5,
                random_state=42,
                scoring='roc_auc'
            )
            
            importance_scores = {}
            for i, feature in enumerate(self.metadata.features):
                importance = result.importances_mean[i]
                # Adjust by business impact
                business_weight = self.metadata.business_impact.get(feature, 0.1)
                adjusted_importance = importance * (1 + business_weight)
                importance_scores[feature] = max(0, float(adjusted_importance))
            
            # Normalize to sum to 1
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                for feature in importance_scores:
                    importance_scores[feature] /= total_importance
            
            logger.info("Global feature importance calculated using permutation importance")
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Could not calculate permutation importance: {e}")
            # Fallback to model built-in importance
            return self._fallback_global_importance()
    
    def _fallback_global_importance(self) -> Dict[str, float]:
        """Fallback global importance calculation."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            # Uniform distribution as last resort
            importances = np.ones(len(self.metadata.features)) / len(self.metadata.features)
        
        # Apply business weights
        importance_scores = {}
        for i, feature in enumerate(self.metadata.features):
            base_importance = float(importances[i])
            business_weight = self.metadata.business_impact.get(feature, 0.1)
            importance_scores[feature] = base_importance * (1 + business_weight)
        
        # Normalize
        total = sum(importance_scores.values())
        if total > 0:
            for feature in importance_scores:
                importance_scores[feature] /= total
        
        return importance_scores
    
    def _generate_visual_summary(self, explanations: Dict[str, Any], 
                               instance: Dict[str, Any]) -> Optional[str]:
        """Generate a comprehensive visual summary of the explanation."""
        try:
            # Create a multi-panel visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('SHAP Force Plot', 'LIME Feature Contributions', 
                              'Feature Importance Ranking', 'Counterfactual Path'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                      [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # SHAP Force Plot (simplified)
            if 'shap' in explanations:
                shap_data = explanations['shap']
                shap_values = np.array(shap_data['shap_values'])
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(1, -1)
                
                # Add positive and negative contributions
                positive_features = shap_values[0][shap_values[0] > 0]
                negative_features = shap_values[0][shap_values[0] < 0]
                
                fig.add_trace(
                    go.Scatter(x=positive_features, y=np.arange(len(positive_features)),
                             mode='markers', name='Risk ↑', marker=dict(color='red')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=negative_features, y=np.arange(len(negative_features)),
                             mode='markers', name='Risk ↓', marker=dict(color='green')),
                    row=1, col=1
                )
            
            # LIME Bar Chart
            if 'lime' in explanations:
                lime_data = explanations['lime']
                features = [seg[0] for seg in lime_data['lime_segments']]
                values = [seg[1] for seg in lime_data['lime_segments']]
                
                colors = ['red' if v > 0 else 'green' for v in values]
                fig.add_trace(
                    go.Bar(x=values, y=features, orientation='h',
                         marker_color=colors, name='LIME Contributions'),
                    row=1, col=2
                )
            
            # Global Feature Importance
            if 'global_importance' in explanations:
                global_imp = explanations['global_importance']
                sorted_features = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
                feat_names = [f[0] for f in sorted_features[:8]]
                feat_values = [f[1] for f in sorted_features[:8]]
                
                fig.add_trace(
                    go.Bar(x=feat_values, y=feat_names, orientation='h',
                         marker_color='purple', name='Global Importance'),
                    row=2, col=1
                )
            
            # Counterfactual Path
            if 'counterfactual' in explanations:
                cf_data = explanations['counterfactual']
                if cf_data:
                    changes = [cf['change_magnitude'] for cf in cf_data[:3]]
                    fig.add_trace(
                        go.Scatter(x=changes, y=[f"CF{i+1}" for i in range(len(changes))],
                                 mode='markers+lines', name='Counterfactual Changes',
                                 marker=dict(color='orange', size=10)),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title=f"Fraud Detection Explanation: {instance.get('transaction_id', 'Unknown')} - "
                      f"Predicted: {'Fraud' if instance_array[0].sum() > 0.5 else 'Legitimate'}",
                height=800,
                showlegend=True
            )
            
            # Convert to base64 (simplified - in practice, use plotly.io.to_image)
            # For this example, return plot configuration
            visual_data = {
                'plotly_config': fig.to_dict(),
                'description': 'Interactive explanation dashboard with SHAP, LIME, and counterfactuals',
                'format': 'interactive_html'
            }
            
            return json.dumps(visual_data)
            
        except Exception as e:
            logger.warning(f"Could not generate visual summary: {e}")
            return None
    
    def _generate_minimal_explanation(self, instance: Dict[str, Any], 
                                   prediction_label: str, confidence: float) -> FraudExplanation:
        """Generate minimal fallback explanation."""
        instance_array = np.array([[
            instance.get(feature, 0) for feature in self.metadata.features
        ]])
        
        prediction = self.model.predict_proba(instance_array)[:, 1][0] if hasattr(self.model, 'predict_proba') else 0.5
        
        # Simple uniform importance as fallback
        feature_importance = {
            feature: 1.0 / len(self.metadata.features) 
            for feature in self.metadata.features
        }
        
        minimal_explanation = FraudExplanation(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            feature_importance=feature_importance,
            regulatory_compliance={reg: {'met': False, 'required': [], 'provided': []} 
                                 for reg in self.compliance_checker.regulations}
        )
        
        self.explanation_history.append(minimal_explanation)
        return minimal_explanation
    
    def _update_performance_metrics(self, explanation: FraudExplanation):
        """Update internal performance metrics for explanation quality."""
        if not self.performance_metrics:
            self.performance_metrics = {
                'total_explanations': 0,
                'high_confidence_count': 0,
                'fraud_predictions': 0,
                'compliance_rate': 0.0,
                'avg_feature_coverage': 0.0,
                'explanation_types': {}
            }
        
        self.performance_metrics['total_explanations'] += 1
        if explanation.confidence > 0.8:
            self.performance_metrics['high_confidence_count'] += 1
        
        if explanation.prediction_label == 'fraud':
            self.performance_metrics['fraud_predictions'] += 1
        
        # Update explanation type usage
        exp_type = explanation.explanation_type.value
        self.performance_metrics['explanation_types'][exp_type] = (
            self.performance_metrics['explanation_types'].get(exp_type, 0) + 1
        )
    
    def generate_batch_explanations(self, instances: List[Dict[str, Any]], 
                                 explanation_type: ExplanationType = ExplanationType.SHAP,
                                 batch_size: int = 10) -> List[FraudExplanation]:
        """Generate explanations for a batch of instances efficiently."""
        explanations = []
        
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(instances)-1)//batch_size + 1}")
            
            for instance in batch:
                try:
                    if explanation_type == ExplanationType.SHAP:
                        exp = self.shap_explainer.explain_instance(
                            np.array([[instance.get(f, 0) for f in self.metadata.features]])
                        )
                        # Convert to full explanation format
                        full_exp = FraudExplanation(
                            prediction=exp['prediction'],
                            prediction_label='fraud' if exp['prediction'] > 0.5 else 'legitimate',
                            confidence=max(exp['prediction'], 1 - exp['prediction']),
                            explanation_type=explanation_type,
                            feature_importance=exp['business_importance']
                        )
                    else:
                        full_exp = self.generate_comprehensive_explanation(
                            instance, [explanation_type]
                        )
                    
                    explanations.append(full_exp)
                    
                except Exception as e:
                    logger.error(f"Failed to explain instance {instance.get('id', i)}: {e}")
                    # Add minimal explanation
                    instance_array = np.array([[instance.get(f, 0) for f in self.metadata.features]])
                    pred = self.model.predict_proba(instance_array)[:, 1][0]
                    minimal = FraudExplanation(
                        prediction=pred,
                        prediction_label='fraud' if pred > 0.5 else 'legitimate',
                        confidence=max(pred, 1-pred),
                        explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                        feature_importance={f: 1/len(self.metadata.features) for f in self.metadata.features}
                    )
                    explanations.append(minimal)
        
        logger.info(f"Generated {len(explanations)} batch explanations")
        return explanations
    
    def export_explanation_report(self, explanation: FraudExplanation, 
                                filepath: str, format_type: str = 'html') -> bool:
        """Export a complete explanation report for audit/compliance."""
        try:
            report_data = {
                'model_information': {
                    'name': self.metadata.model_name,
                    'version': self.metadata.version,
                    'type': self.metadata.model_type,
                    'features': self.metadata.features
                },
                'prediction': {
                    'score': explanation.prediction,
                    'label': explanation.prediction_label,
                    'confidence': explanation.confidence,
                    'timestamp': explanation.timestamp
                },
                'explanation': {
                    'type': explanation.explanation_type.value,
                    'feature_importance': explanation.feature_importance,
                    'global_importance': explanation.global_importance,
                    'counterfactuals': explanation.counterfactuals,
                    'business_context': self._add_business_context(explanation)
                },
                'compliance': explanation.regulatory_compliance,
                'audit_information': {
                    'generated_by': 'FraudXAI System',
                    'explanation_id': f"exp_{int(explanation.timestamp)}",
                    'export_timestamp': time.time()
                }
            }
            
            if format_type == 'html':
                self._generate_html_report(report_data, filepath)
            elif format_type == 'json':
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            elif format_type == 'pdf':
                # Would require additional PDF generation library
                logger.warning("PDF export not implemented in this version")
                return False
            
            logger.info(f"Explanation report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export explanation report: {e}")
            return False
    
    def _add_business_context(self, explanation: FraudExplanation) -> Dict[str, Any]:
        """Add business context and impact analysis to the explanation."""
        context = {
            'risk_category': 'high' if explanation.prediction > 0.8 else 
                           'medium' if explanation.prediction > 0.5 else 'low',
            'business_impact': self._calculate_business_impact(explanation),
            'recommended_actions': self._generate_business_actions(explanation),
            'regulatory_flags': self._identify_regulatory_flags(explanation)
        }
        
        return context
    
    def _calculate_business_impact(self, explanation: FraudExplanation) -> Dict[str, float]:
        """Calculate the business impact of the fraud prediction."""
        impact_scores = {}
        
        # Transaction risk impact
        if 'transaction_amount' in explanation.feature_importance:
            amount_impact = explanation.feature_importance['transaction_amount']
            impact_scores['financial_exposure'] = amount_impact * 10000  # Scale to currency
        
        # Account risk impact
        if 'account_age_days' in explanation.feature_importance:
            age_risk = 1.0 / (explanation.feature_importance.get('account_age_days', 1) + 1)
            impact_scores['account_compromise_risk'] = age_risk * 0.8
        
        # Velocity risk
        if 'velocity_count' in explanation.feature_importance:
            velocity_impact = explanation.feature_importance['velocity_count']
            impact_scores['velocity_risk'] = min(1.0, velocity_impact * 5)
        
        return impact_scores
    
    def _generate_business_actions(self, explanation: FraudExplanation) -> List[str]:
        """Generate recommended business actions based on explanation."""
        actions = []
        prediction = explanation.prediction
        
        if prediction > 0.8:  # High risk
            actions.extend([
                "IMMEDIATE: Block transaction and freeze account",
                "NOTIFY: Alert fraud team and relationship manager",
                "INVESTIGATE: Initiate full case investigation",
                "COMPLY: File SAR (Suspicious Activity Report) within 24 hours"
            ])
        elif prediction > 0.5:  # Medium risk
            actions.extend([
                "VERIFY: Request additional customer verification",
                "LIMIT: Apply transaction limits and monitoring",
                "REVIEW: Queue for manual review within 1 hour",
                "MONITOR: Enable enhanced behavioral monitoring"
            ])
        else:  # Low risk
            actions.extend([
                "APPROVE: Allow transaction with standard monitoring",
                "LOG: Record for pattern analysis and model training",
                "CONTINUE: Maintain normal risk assessment cadence"
            ])
        
        # Add feature-specific actions
        top_features = sorted(explanation.feature_importance.items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for feature, importance in top_features:
            if abs(importance) > 0.1:
                if 'geolocation' in feature and importance > 0:
                    actions.append("GEO: Verify unusual location with customer")
                elif 'device' in feature and importance > 0:
                    actions.append("DEVICE: Validate new device registration")
                elif 'velocity' in feature:
                    actions.append("VELOCITY: Implement temporary velocity controls")
        
        return actions
    
    def _identify_regulatory_flags(self, explanation: FraudExplanation) -> List[str]:
        """Identify regulatory flags based on explanation content."""
        flags = []
        
        # Check for high-risk indicators
        if explanation.prediction > 0.9:
            flags.append("HIGH_RISK_DECISION: Requires enhanced regulatory reporting")
        
        # Check for sensitive feature usage
        sensitive_features = ['ssn', 'account_number', 'pii', 'sensitive']
        for feature in explanation.feature_importance:
            if any(sens in feature.lower() for sens in sensitive_features):
                if abs(explanation.feature_importance[feature]) > 0.05:
                    flags.append(f"SENSITIVE_FEATURE: {feature} has significant influence")
                    break
        
        # Check compliance validation
        compliance = explanation.regulatory_compliance
        for regulation, status in compliance.items():
            if not status.get('complete', False):
                flags.append(f"INCOMPLETE_{regulation}: Missing required explanation elements")
        
        return flags
    
    def _generate_html_report(self, report_data: Dict[str, Any], filepath: str):
        """Generate HTML report (simplified version)."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Explanation Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .prediction {{ font-size: 24px; font-weight: bold; color: {'red' if report_data['prediction']['score'] > 0.5 else 'green'}; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #ddd; }}
                .high-risk {{ background: #ffebee; border-left-color: #f44336; }}
                .medium-risk {{ background: #fff3e0; border-left-color: #ff9800; }}
                .low-risk {{ background: #e8f5e8; border-left-color: #4caf50; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f2f2f2; }}
                .compliance-warning {{ color: #f44336; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fraud Detection Explanation Report</h1>
                <p><strong>Model:</strong> {report_data['model_information']['name']} v{report_data['model_information']['version']}</p>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report_data['audit_information']['export_timestamp']))}</p>
            </div>
            
            <div class="section {'high-risk' if report_data['prediction']['score'] > 0.8 else 'medium-risk' if report_data['prediction']['score'] > 0.5 else 'low-risk'}">
                <h2>Prediction Result</h2>
                <p class="prediction">{report_data['prediction']['label'].upper()}</p>
                <p><strong>Confidence:</strong> {report_data['prediction']['confidence']:.1%}</p>
                <p><strong>Raw Score:</strong> {report_data['prediction']['score']:.4f}</p>
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                <table>
                    <tr><th>Feature</th><th>Importance</th><th>Contribution</th><th>Business Impact</th></tr>
                    {''.join([f'<tr><td>{feat}</td><td>{imp:.4f}</td><td>{"+" if contrib > 0 else ""}{contrib:.4f}</td><td>{"High" if bi > 0.15 else "Medium" if bi > 0.05 else "Low"}</td></tr>' 
                            for feat, imp in report_data['explanation']['feature_importance'].items() 
                            for contrib, bi in [(0, report_data['model_information']['business_impact'].get(feat, 0.1))]])}
                </table>
            </div>
            
            {self._compliance_html_section(report_data['compliance'])}
            
            <div class="section">
                <h2>Recommended Actions</h2>
                <ul>
                    {''.join([f'<li>{action}</li>' for action in report_data['explanation']['business_context']['recommended_actions']])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Audit Information</h2>
                <p><strong>Explanation ID:</strong> {report_data['audit_information']['explanation_id']}</p>
                <p><strong>Generated by:</strong> {report_data['audit_information']['generated_by']}</p>
                {''.join([f'<p class="compliance-warning"><strong>{reg}:</strong> {"Compliant" if status["complete"] else "Non-compliant - ' + ', '.join(status["missing"])}</p>' 
                        for reg, status in report_data['compliance'].items()])}
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_template)
    
    def _compliance_html_section(self, compliance_data: Dict[str, Dict[str, Any]]) -> str:
        """Generate HTML section for compliance information."""
        html = '<div class="section">\n<h2>Regulatory Compliance</h2>\n<table>\n'
        html += '<tr><th>Regulation</th><th>Status</th><th>Required</th><th>Provided</th></tr>\n'
        
        for regulation, status in compliance_data.items():
            status_class = "complete" if status['complete'] else "incomplete"
            html += f'<tr><td>{regulation}</td><td class="{status_class}">{"Compliant" if status["complete"] else "Non-compliant"}</td>'
            html += f'<td>{", ".join(status["required"])}</td><td>{", ".join(status["provided"])}</td></tr>\n'
        
        html += '</table></div>'
        return html
    
    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated explanations."""
        if not self.explanation_history:
            return {'total_explanations': 0, 'insights': []}
        
        stats = {
            'total_explanations': len(self.explanation_history),
            'fraud_predictions': sum(1 for e in self.explanation_history if e.prediction_label == 'fraud'),
            'high_confidence': sum(1 for e in self.explanation_history if e.confidence > 0.8),
            'avg_confidence': np.mean([e.confidence for e in self.explanation_history]),
            'most_influential_features': self._analyze_feature_influence(),
            'compliance_summary': self.compliance_checker.audit_log[-10:] if self.compliance_checker.audit_log else [],
            'model_performance': self.performance_metrics,
            'insights': self._generate_analytics_insights()
        }
        
        return stats
    
    def _analyze_feature_influence(self) -> Dict[str, float]:
        """Analyze which features most frequently drive decisions."""
        feature_usage = {}
        
        for explanation in self.explanation_history[-100:]:  # Last 100 explanations
            for feature, importance in explanation.feature_importance.items():
                if abs(importance) > 0.05:  # Significant influence
                    feature_usage[feature] = feature_usage.get(feature, 0) + abs(importance)
        
        # Normalize by total explanations
        total_influence = sum(feature_usage.values())
        if total_influence > 0:
            for feature in feature_usage:
                feature_usage[feature] /= total_influence
        
        return dict(sorted(feature_usage.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_analytics_insights(self) -> List[str]:
        """Generate actionable insights from explanation patterns."""
        insights = []
        fraud_count = sum(1 for e in self.explanation_history if e.prediction_label == 'fraud')
        total_count = len(self.explanation_history)
        
        if total_count > 0:
            fraud_rate = fraud_count / total_count
            if fraud_rate > 0.1:
                insights.append(f"HIGH FRAUD RATE: {fraud_rate:.1%} of predictions are fraud - review model thresholds")
            elif fraud_rate < 0.01:
                insights.append(f"LOW FRAUD DETECTION: Only {fraud_rate:.1%} fraud predictions - potential under-detection")
        
        # Feature dominance
        if self.explanation_history:
            all_features = []
            for exp in self.explanation_history[-50:]:
                top_feature = max(exp.feature_importance.items(), key=lambda x: abs(x[1]))
                all_features.append(top_feature[0])
            
            from collections import Counter
            feature_counter = Counter(all_features)
            dominant_feature = feature_counter.most_common(1)[0]
            
            if dominant_feature[1] > len(all_features) * 0.3:
                insights.append(f"FEATURE DOMINANCE: {dominant_feature[0]} drives {dominant_feature[1]/len(all_features):.0%} of decisions - validate business logic")
        
        # Compliance issues
        recent_compliance = [log for log in self.compliance_checker.audit_log[-20:]]
        non_compliant = sum(1 for log in recent_compliance if not log['compliance_status'].get('compliant', True))
        
        if non_compliant > len(recent_compliance) * 0.2:
            insights.append(f"COMPLIANCE GAP: {non_compliant/len(recent_compliance):.0%} of recent explanations non-compliant")
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for fraud detection
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic fraud detection dataset
    data = {
        'transaction_amount': np.random.lognormal(5, 1, n_samples),
        'account_age_days': np.random.exponential(365, n_samples),
        'velocity_count': np.random.poisson(2, n_samples),
        'geolocation_distance': np.abs(np.random.normal(0, 500, n_samples)),
        'device_fingerprint_score': np.random.beta(2, 5, n_samples),
        'behavioral_anomaly': np.random.beta(1, 5, n_samples),
        'ip_reputation': np.random.beta(5, 2, n_samples),
        'time_of_day': np.random.uniform(0, 24, n_samples),
        'email_domain_age': np.random.exponential(365*2, n_samples),
        'merchant_category': np.random.choice(['retail', 'finance', 'tech', 'travel'], n_samples)
    }
    
    # One-hot encode categorical features
    merchant_encoded = pd.get_dummies(pd.Series(data['merchant_category']))
    for col in merchant_encoded.columns:
        data[col] = merchant_encoded[col].values
    
    df = pd.DataFrame(data)
    
    # Generate target variable with realistic fraud patterns
    fraud_probability = (
        0.3 * (df['transaction_amount'] > 10000) +
        0.2 * (df['account_age_days'] < 30) +
        0.25 * (df['velocity_count'] > 5) +
        0.15 * (df['geolocation_distance'] > 1000) +
        0.1 * (df['behavioral_anomaly'] > 0.8) +
        0.05 * (df['ip_reputation'] < 0.2) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['fraud'] = (fraud_probability > 0.5).astype(int)
    df['fraud_probability'] = np.clip(fraud_probability, 0, 1)
    
    # Train simple fraud detection model
    from sklearn.model_selection import train_test_split
    
    feature_cols = [col for col in df.columns if col != 'fraud' and col != 'fraud_probability']
    X = df[feature_cols]
    y = df['fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])
    test_score = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    print(f"Model Performance - Train AUC: {train_score:.3f}, Test AUC: {test_score:.3f}")
    
    # Create model metadata
    metadata = ModelMetadata(
        model_name="Fraud Detection RF",
        model_type="Random Forest",
        features=feature_cols,
        target_name="fraud"
    )
    
    # Add feature descriptions
    metadata.add_feature_description(
        'transaction_amount', 
        "Transaction amount in USD", 
        business_critical=True,
        compliance_tags=['financial']
    )
    metadata.add_feature_description(
        'account_age_days', 
        "Age of the account in days",
        business_critical=True
    )
    metadata.add_feature_description(
        'velocity_count', 
        "Number of transactions in last 24 hours",
        business_critical=True,
        compliance_tags=['velocity']
    )
    metadata.add_feature_description(
        'geolocation_distance', 
        "Distance from account's home location in km",
        business_critical=True
    )
    metadata.set_feature_range('transaction_amount', 0, 100000, 'USD')
    metadata.set_feature_range('account_age_days', 0, 3650, 'days')
    metadata.set_feature_range('velocity_count', 0, 50, 'transactions')
    metadata.set_feature_range('geolocation_distance', 0, 20000, 'km')
    
    # Initialize XAI system
    xai_system = FraudXAI(
        model=rf_model,
        model_metadata=metadata,
        training_data=X_train,
        background_data=X_train.values,
        jurisdiction="US"
    )
    
    # Generate explanation for a high-risk transaction
    high_risk_instance = {
        'transaction_id': 'TXN_12345',
        'transaction_amount': 25000,
        'account_age_days': 15,
        'velocity_count': 8,
        'geolocation_distance': 2500,
        'device_fingerprint_score': 0.3,
        'behavioral_anomaly': 0.75,
        'ip_reputation': 0.25,
        'time_of_day': 2.5,
        'email_domain_age': 30,
        'merchant_category': 'finance'
    }
    
    # Generate comprehensive explanation
    explanation = xai_system.generate_comprehensive_explanation(
        high_risk_instance,
        explanation_types=[ExplanationType.SHAP, ExplanationType.COUNTERFACTUAL, ExplanationType.FEATURE_IMPORTANCE],
        target_class='legitimate'
    )
    
    print("\n" + "="*60)
    print("FRAUD EXPLANATION REPORT")
    print("="*60)
    print(f"Prediction: {explanation.prediction_label} (Confidence: {explanation.confidence:.1%})")
    print(f"Raw Score: {explanation.prediction:.4f}")
    print("\nTop Risk Factors:")
    
    for feature_data in explanation.feature_importance.get('top_features', []):
        print(f"  • {feature_data['feature']}: {feature_data['shap_value']:.4f} "
              f"({feature_data['direction']}) - {feature_data['description']}")
    
    if explanation.counterfactuals:
        print("\nCounterfactual Actions to Avoid Fraud:")
        for cf in explanation.counterfactuals[:2]:
            print(f"  • Change {cf['changed_feature']} from {cf['original_value']:.1f} to {cf['new_value']:.1f} "
                  f"(distance: {cf['distance']:.3f})")
    
    print(f"\nCompliance Status: {'PASS' if all(s['complete'] for s in explanation.regulatory_compliance.values()) else 'REVIEW NEEDED'}")
    
    # Export report
    xai_system.export_explanation_report(explanation, "fraud_explanation_report.html", "html")
    
    # Generate batch explanations for analysis
    test_instances = X_test.iloc[:5].to_dict('records')
    batch_explanations = xai_system.generate_batch_explanations(test_instances)
    
    print(f"\nBatch processing completed: {len(batch_explanations)} explanations generated")
    
    # Get system statistics
    stats = xai_system.get_explanation_statistics()
    print(f"\nXAI System Stats:")
    print(f"  Total explanations: {stats['total_explanations']}")
    print(f"  Fraud predictions: {stats['fraud_predictions']}")
    print(f"  High confidence: {stats['high_confidence']}")
    print(f"  Average confidence: {stats['avg_confidence']:.1%}")
    
    if stats['insights']:
        print("\nKey Insights:")
        for insight in stats['insights']:
            print(f"  • {insight}")
