"""
Voice Biometric Authentication Models
Implements voice authentication, deepfake detection, and emotional stress analysis
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

@dataclass
class VoiceFeatures:
    """Extracted voice features"""
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    pitch: np.ndarray
    energy: np.ndarray
    zero_crossing_rate: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    chroma: np.ndarray
    mel_spectrogram: np.ndarray
    duration: float
    sample_rate: int

@dataclass
class VoiceEmbedding:
    """Voice embedding vector"""
    user_id: str
    embedding: np.ndarray
    confidence: float
    sample_count: int
    created_at: datetime
    updated_at: datetime
    language: str
    audio_quality_score: float

@dataclass
class VoiceAuthenticationResult:
    """Voice authentication result"""
    user_id: str
    is_authentic: bool
    similarity_score: float
    confidence: float
    deepfake_probability: float
    stress_level: float
    audio_quality: float
    reasons: List[str]
    timestamp: datetime
    processing_time_ms: float

@dataclass
class DeepfakeDetectionResult:
    """Deepfake detection result"""
    is_deepfake: bool
    deepfake_probability: float
    confidence: float
    detection_method: str
    artifacts_detected: List[str]
    spectral_anomalies: List[str]
    timestamp: datetime

class VoiceFeatureExtractor:
    """Extract features from audio for voice biometrics"""
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_features(self, audio: np.ndarray) -> VoiceFeatures:
        """Extract comprehensive voice features from audio"""
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # Extract pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        pitch = np.array([pitches[:, i][magnitudes[:, i].argmax()] 
                         for i in range(pitches.shape[1])])
        
        # Extract energy
        energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length
        )[0]
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        duration = len(audio) / self.sample_rate
        
        return VoiceFeatures(
            mfcc=mfcc,
            pitch=pitch,
            energy=energy,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            chroma=chroma,
            mel_spectrogram=mel_spec,
            duration=duration,
            sample_rate=self.sample_rate
        )
    
    def extract_statistical_features(self, features: VoiceFeatures) -> Dict[str, float]:
        """Extract statistical features from voice features"""
        stats = {}
        
        # MFCC statistics
        for i in range(features.mfcc.shape[0]):
            mfcc_coef = features.mfcc[i, :]
            stats[f'mfcc_{i}_mean'] = float(np.mean(mfcc_coef))
            stats[f'mfcc_{i}_std'] = float(np.std(mfcc_coef))
            stats[f'mfcc_{i}_max'] = float(np.max(mfcc_coef))
            stats[f'mfcc_{i}_min'] = float(np.min(mfcc_coef))
        
        # Pitch statistics
        valid_pitch = features.pitch[features.pitch > 0]
        if len(valid_pitch) > 0:
            stats['pitch_mean'] = float(np.mean(valid_pitch))
            stats['pitch_std'] = float(np.std(valid_pitch))
            stats['pitch_max'] = float(np.max(valid_pitch))
            stats['pitch_min'] = float(np.min(valid_pitch))
            stats['pitch_range'] = stats['pitch_max'] - stats['pitch_min']
        else:
            stats['pitch_mean'] = 0.0
            stats['pitch_std'] = 0.0
            stats['pitch_max'] = 0.0
            stats['pitch_min'] = 0.0
            stats['pitch_range'] = 0.0
        
        # Energy statistics
        stats['energy_mean'] = float(np.mean(features.energy))
        stats['energy_std'] = float(np.std(features.energy))
        stats['energy_max'] = float(np.max(features.energy))
        
        # Zero crossing rate statistics
        stats['zcr_mean'] = float(np.mean(features.zero_crossing_rate))
        stats['zcr_std'] = float(np.std(features.zero_crossing_rate))
        
        # Spectral statistics
        stats['spectral_centroid_mean'] = float(np.mean(features.spectral_centroid))
        stats['spectral_centroid_std'] = float(np.std(features.spectral_centroid))
        stats['spectral_rolloff_mean'] = float(np.mean(features.spectral_rolloff))
        stats['spectral_rolloff_std'] = float(np.std(features.spectral_rolloff))
        
        # Duration
        stats['duration'] = features.duration
        
        return stats

class VoiceEmbeddingGenerator:
    """Generate voice embeddings using deep learning"""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.model = None
    
    def build_model(self):
        """Build voice embedding model (simplified version)"""
        # In production, use pre-trained models like Resemblyzer or SpeechBrain
        # This is a placeholder for the architecture
        pass
    
    def generate_embedding(self, features: VoiceFeatures) -> np.ndarray:
        """Generate voice embedding from features"""
        # Simplified embedding generation using MFCC statistics
        # In production, use deep learning models
        
        mfcc_mean = np.mean(features.mfcc, axis=1)
        mfcc_std = np.std(features.mfcc, axis=1)
        
        pitch_stats = np.array([
            np.mean(features.pitch),
            np.std(features.pitch),
            np.max(features.pitch),
            np.min(features.pitch)
        ])
        
        energy_stats = np.array([
            np.mean(features.energy),
            np.std(features.energy),
            np.max(features.energy)
        ])
        
        spectral_stats = np.array([
            np.mean(features.spectral_centroid),
            np.std(features.spectral_centroid),
            np.mean(features.spectral_rolloff),
            np.std(features.spectral_rolloff)
        ])
        
        # Concatenate all features
        embedding = np.concatenate([
            mfcc_mean,
            mfcc_std,
            pitch_stats,
            energy_stats,
            spectral_stats
        ])
        
        # Pad or truncate to desired dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def compare_embeddings(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compare two voice embeddings using cosine similarity"""
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        )
        return float(similarity)

class DeepfakeDetector:
    """Detect deepfake/synthetic voice"""
    
    def __init__(self):
        self.model = None
        self.threshold = 0.5
    
    def detect_deepfake(self, features: VoiceFeatures) -> DeepfakeDetectionResult:
        """Detect if voice is deepfake/synthetic"""
        artifacts = []
        spectral_anomalies = []
        
        # Check for spectral artifacts common in deepfakes
        mel_spec = features.mel_spectrogram
        
        # 1. Check for unnatural spectral patterns
        spectral_variance = np.var(mel_spec, axis=1)
        if np.max(spectral_variance) / (np.mean(spectral_variance) + 1e-8) > 10:
            spectral_anomalies.append('High spectral variance inconsistency')
        
        # 2. Check for pitch artifacts
        pitch_jumps = np.abs(np.diff(features.pitch))
        if np.max(pitch_jumps) > 100:  # Unnatural pitch jumps
            artifacts.append('Unnatural pitch discontinuities')
        
        # 3. Check for energy artifacts
        energy_variance = np.var(features.energy)
        if energy_variance < 0.001:  # Too consistent energy
            artifacts.append('Unnaturally consistent energy levels')
        
        # 4. Check for high-frequency artifacts
        high_freq_energy = np.mean(mel_spec[-10:, :])
        low_freq_energy = np.mean(mel_spec[:10, :])
        if high_freq_energy / (low_freq_energy + 1e-8) > 2:
            spectral_anomalies.append('Unusual high-frequency content')
        
        # 5. Check for phase inconsistencies (simplified)
        phase_consistency = self._check_phase_consistency(features)
        if phase_consistency < 0.5:
            artifacts.append('Phase inconsistencies detected')
        
        # Calculate deepfake probability
        artifact_score = len(artifacts) / 5.0
        spectral_score = len(spectral_anomalies) / 4.0
        deepfake_probability = (artifact_score + spectral_score) / 2.0
        
        is_deepfake = deepfake_probability >= self.threshold
        confidence = abs(deepfake_probability - 0.5) * 2  # Distance from decision boundary
        
        return DeepfakeDetectionResult(
            is_deepfake=is_deepfake,
            deepfake_probability=deepfake_probability,
            confidence=confidence,
            detection_method='spectral_analysis',
            artifacts_detected=artifacts,
            spectral_anomalies=spectral_anomalies,
            timestamp=datetime.now()
        )
    
    def _check_phase_consistency(self, features: VoiceFeatures) -> float:
        """Check phase consistency (simplified)"""
        # In production, use STFT phase analysis
        # This is a simplified version
        zcr_variance = np.var(features.zero_crossing_rate)
        consistency = 1.0 / (1.0 + zcr_variance * 100)
        return consistency

class EmotionalStressAnalyzer:
    """Analyze emotional stress from voice"""
    
    def __init__(self):
        self.baseline_pitch_range = (80, 250)  # Hz
        self.baseline_energy_std = 0.1
    
    def analyze_stress(self, features: VoiceFeatures) -> float:
        """Analyze stress level from voice features (0-1 scale)"""
        stress_indicators = []
        
        # 1. Pitch analysis
        valid_pitch = features.pitch[features.pitch > 0]
        if len(valid_pitch) > 0:
            pitch_mean = np.mean(valid_pitch)
            pitch_std = np.std(valid_pitch)
            
            # Higher pitch and variance indicate stress
            pitch_stress = (pitch_mean - self.baseline_pitch_range[0]) / (
                self.baseline_pitch_range[1] - self.baseline_pitch_range[0]
            )
            pitch_stress = np.clip(pitch_stress, 0, 1)
            
            variance_stress = pitch_std / 50.0  # Normalize
            variance_stress = np.clip(variance_stress, 0, 1)
            
            stress_indicators.append((pitch_stress + variance_stress) / 2)
        
        # 2. Energy analysis
        energy_std = np.std(features.energy)
        energy_stress = energy_std / self.baseline_energy_std
        energy_stress = np.clip(energy_stress, 0, 1)
        stress_indicators.append(energy_stress)
        
        # 3. Speaking rate (from zero crossing rate)
        zcr_mean = np.mean(features.zero_crossing_rate)
        rate_stress = zcr_mean * 10  # Normalize
        rate_stress = np.clip(rate_stress, 0, 1)
        stress_indicators.append(rate_stress)
        
        # 4. Voice tremor (from pitch variations)
        if len(valid_pitch) > 1:
            pitch_tremor = np.mean(np.abs(np.diff(valid_pitch)))
            tremor_stress = pitch_tremor / 20.0  # Normalize
            tremor_stress = np.clip(tremor_stress, 0, 1)
            stress_indicators.append(tremor_stress)
        
        # Calculate overall stress level
        stress_level = np.mean(stress_indicators) if stress_indicators else 0.0
        
        return float(stress_level)

class AudioQualityAssessor:
    """Assess audio quality for voice authentication"""
    
    def __init__(self):
        self.min_duration = 1.0  # seconds
        self.max_duration = 30.0  # seconds
        self.min_sample_rate = 8000
        self.optimal_sample_rate = 16000
    
    def assess_quality(self, audio: np.ndarray, 
                      sample_rate: int,
                      features: VoiceFeatures) -> Tuple[float, List[str]]:
        """Assess audio quality and return score (0-1) and issues"""
        issues = []
        quality_scores = []
        
        # 1. Check duration
        duration = len(audio) / sample_rate
        if duration < self.min_duration:
            issues.append(f'Audio too short: {duration:.1f}s (minimum {self.min_duration}s)')
            quality_scores.append(0.3)
        elif duration > self.max_duration:
            issues.append(f'Audio too long: {duration:.1f}s (maximum {self.max_duration}s)')
            quality_scores.append(0.7)
        else:
            quality_scores.append(1.0)
        
        # 2. Check sample rate
        if sample_rate < self.min_sample_rate:
            issues.append(f'Sample rate too low: {sample_rate}Hz')
            quality_scores.append(0.3)
        elif sample_rate < self.optimal_sample_rate:
            quality_scores.append(0.7)
        else:
            quality_scores.append(1.0)
        
        # 3. Check signal-to-noise ratio
        snr = self._estimate_snr(audio)
        if snr < 10:
            issues.append(f'Low signal-to-noise ratio: {snr:.1f}dB')
            quality_scores.append(0.4)
        elif snr < 20:
            quality_scores.append(0.7)
        else:
            quality_scores.append(1.0)
        
        # 4. Check clipping
        clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
        if clipping_ratio > 0.01:
            issues.append(f'Audio clipping detected: {clipping_ratio*100:.1f}%')
            quality_scores.append(0.5)
        else:
            quality_scores.append(1.0)
        
        # 5. Check silence ratio
        silence_threshold = 0.01
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        if silence_ratio > 0.5:
            issues.append(f'Too much silence: {silence_ratio*100:.1f}%')
            quality_scores.append(0.6)
        else:
            quality_scores.append(1.0)
        
        # 6. Check frequency content
        energy_mean = np.mean(features.energy)
        if energy_mean < 0.01:
            issues.append('Very low energy levels')
            quality_scores.append(0.4)
        else:
            quality_scores.append(1.0)
        
        overall_quality = np.mean(quality_scores)
        
        return float(overall_quality), issues
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        # Simple SNR estimation
        signal_power = np.mean(audio ** 2)
        
        # Estimate noise from low-energy segments
        energy = np.abs(audio)
        noise_threshold = np.percentile(energy, 10)
        noise_segments = audio[energy < noise_threshold]
        
        if len(noise_segments) > 0:
            noise_power = np.mean(noise_segments ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        else:
            snr = 30.0  # Assume good SNR if no noise detected
        
        return float(snr)

class VoiceAuthenticator:
    """Main voice authentication system"""
    
    def __init__(self):
        self.feature_extractor = VoiceFeatureExtractor()
        self.embedding_generator = VoiceEmbeddingGenerator()
        self.deepfake_detector = DeepfakeDetector()
        self.stress_analyzer = EmotionalStressAnalyzer()
        self.quality_assessor = AudioQualityAssessor()
        self.similarity_threshold = 0.75
    
    def enroll_user(self, user_id: str, audio: np.ndarray, 
                   sample_rate: int, language: str = 'en-US') -> VoiceEmbedding:
        """Enroll user with voice sample"""
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Assess quality
        quality_score, quality_issues = self.quality_assessor.assess_quality(
            audio, sample_rate, features
        )
        
        if quality_score < 0.5:
            raise ValueError(f'Audio quality too low: {quality_issues}')
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embedding(features)
        
        return VoiceEmbedding(
            user_id=user_id,
            embedding=embedding,
            confidence=quality_score,
            sample_count=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            language=language,
            audio_quality_score=quality_score
        )
    
    def authenticate(self, user_id: str, audio: np.ndarray,
                    sample_rate: int,
                    enrolled_embedding: VoiceEmbedding) -> VoiceAuthenticationResult:
        """Authenticate user with voice sample"""
        start_time = datetime.now()
        
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Assess quality
        quality_score, quality_issues = self.quality_assessor.assess_quality(
            audio, sample_rate, features
        )
        
        # Detect deepfake
        deepfake_result = self.deepfake_detector.detect_deepfake(features)
        
        # Analyze stress
        stress_level = self.stress_analyzer.analyze_stress(features)
        
        # Generate embedding
        current_embedding = self.embedding_generator.generate_embedding(features)
        
        # Compare embeddings
        similarity_score = self.embedding_generator.compare_embeddings(
            enrolled_embedding.embedding,
            current_embedding
        )
        
        # Determine authentication result
        is_authentic = (
            similarity_score >= self.similarity_threshold and
            not deepfake_result.is_deepfake and
            quality_score >= 0.5
        )
        
        # Calculate confidence
        confidence = min(
            similarity_score,
            1 - deepfake_result.deepfake_probability,
            quality_score
        )
        
        # Generate reasons
        reasons = []
        if similarity_score < self.similarity_threshold:
            reasons.append(f'Voice similarity too low: {similarity_score:.2f}')
        if deepfake_result.is_deepfake:
            reasons.append('Deepfake detected')
            reasons.extend(deepfake_result.artifacts_detected)
        if quality_score < 0.5:
            reasons.append('Poor audio quality')
            reasons.extend(quality_issues)
        if stress_level > 0.7:
            reasons.append(f'High stress level detected: {stress_level:.2f}')
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return VoiceAuthenticationResult(
            user_id=user_id,
            is_authentic=is_authentic,
            similarity_score=similarity_score,
            confidence=confidence,
            deepfake_probability=deepfake_result.deepfake_probability,
            stress_level=stress_level,
            audio_quality=quality_score,
            reasons=reasons,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
