/**
 * Unit Tests for AdvancedRiskEngine
 * Comprehensive testing of behavioral signal processing, risk calculation, and anomaly detection
 * Tests cover normalization, profile updates, weighted scoring, Z-score anomalies, and edge cases
 */

import { AdvancedRiskEngine, BehavioralSignal, RiskFactors } from '../../../services/biometric-service/src/advanced-risk-engine';

// Test utilities and mock data
const mockSignals: BehavioralSignal[] = [
  {
    keystrokeVariance: 0.15,
    mouseEntropy: 2.3,
    touchPressureStd: 0.08,
    deviceStability: 0.92,
    sessionDuration: 1200, // 20 minutes
    locationVelocity: 45.2 // km/h
  },
  {
    keystrokeVariance: 0.22,
    mouseEntropy: 1.8,
    touchPressureStd: 0.12,
    deviceStability: 0.85,
    sessionDuration: 1800, // 30 minutes
    locationVelocity: 120.5 // km/h
  },
  {
    keystrokeVariance: 0.10,
    mouseEntropy: 2.7,
    touchPressureStd: 0.05,
    deviceStability: 0.95,
    sessionDuration: 900, // 15 minutes
    locationVelocity: 12.3 // km/h
  }
];

const anomalousSignal: BehavioralSignal = {
  keystrokeVariance: 0.85, // Extremely high variance (potential bot)
  mouseEntropy: 0.1, // Very low entropy (scripted movement)
  touchPressureStd: 0.45, // Inconsistent pressure
  deviceStability: 0.2, // Frequent device changes
  sessionDuration: 60, // Very short session
  locationVelocity: 850.0 // Impossible speed (airplane+)
};

const normalSignal: BehavioralSignal = {
  keystrokeVariance: 0.12,
  mouseEntropy: 2.1,
  touchPressureStd: 0.07,
  deviceStability: 0.90,
  sessionDuration: 1500,
  locationVelocity: 25.0
};

describe('AdvancedRiskEngine', () => {
  let engine: AdvancedRiskEngine;
  let sessionId: string;

  beforeEach(() => {
    sessionId = `test_session_${Date.now()}`;
    engine = new AdvancedRiskEngine(sessionId);
  });

  afterEach(() => {
    // Clean up any test artifacts
    jest.clearAllMocks();
  });

  describe('Constructor and Initialization', () => {
    it('should initialize with correct thresholds', () => {
      expect(engine).toBeInstanceOf(AdvancedRiskEngine);
      expect(engine['THRESHOLDS']).toEqual({
        HIGH_RISK: 0.85,
        MEDIUM_RISK: 0.65,
        LOW_RISK: 0.35,
        BEHAVIORAL_WEIGHT: 0.4,
        DEVICE_WEIGHT: 0.25,
        CONTEXTUAL_WEIGHT: 0.2,
        TEMPORAL_WEIGHT: 0.15
      });
      expect(engine['userProfile']).toBeInstanceOf(Map);
      expect(engine['sessionHistory']).toEqual([]);
    });

    it('should handle multiple engines with different session IDs', () => {
      const engine2 = new AdvancedRiskEngine('different_session');
      expect(engine2).not.toBe(engine);
      expect(engine2['sessionId']).not.toBe(sessionId);
    });
  });

  describe('Signal Ingestion and Normalization', () => {
    it('should correctly normalize behavioral signals to 0-1 range', () => {
      const signal = mockSignals[0];
      const normalized = engine['normalizeSignal'](signal);
      
      expect(normalized.keystrokeVariance).toBeGreaterThanOrEqual(0);
      expect(normalized.keystrokeVariance).toBeLessThanOrEqual(1);
      expect(normalized.mouseEntropy).toBeGreaterThanOrEqual(0);
      expect(normalized.mouseEntropy).toBeLessThanOrEqual(1);
      expect(normalized.touchPressureStd).toBeGreaterThanOrEqual(0);
      expect(normalized.touchPressureStd).toBeLessThanOrEqual(1);
      
      // Check that values are scaled appropriately
      expect(normalized.deviceStability).toBeCloseTo(0.92, 2); // Already in range
      expect(normalized.sessionDuration).toBeLessThan(1); // Should be normalized from seconds
      expect(normalized.locationVelocity).toBeLessThan(1); // Should be normalized from km/h
    });

    it('should handle edge cases in normalization (zero/negative values)', () => {
      const edgeSignal: BehavioralSignal = {
        keystrokeVariance: 0,
        mouseEntropy: -0.1, // Invalid negative
        touchPressureStd: 0,
        deviceStability: 0,
        sessionDuration: 0,
        locationVelocity: 0
      };
      
      const normalized = engine['normalizeSignal'](edgeSignal);
      
      expect(normalized.keystrokeVariance).toBe(0);
      expect(normalized.mouseEntropy).toBe(0); // Negative clamped to 0
      expect(normalized.touchPressureStd).toBe(0);
      expect(normalized.deviceStability).toBe(0);
      expect(normalized.sessionDuration).toBe(0);
      expect(normalized.locationVelocity).toBe(0);
    });

    it('should handle extreme values in normalization (very large numbers)', () => {
      const extremeSignal: BehavioralSignal = {
        keystrokeVariance: 1000,
        mouseEntropy: 50,
        touchPressureStd: 10,
        deviceStability: 2, // Invalid >1
        sessionDuration: 86400000, // 24 hours in ms
        locationVelocity: 10000 // Supersonic speed
      };
      
      const normalized = engine['normalizeSignal'](extremeSignal);
      
      expect(normalized.keystrokeVariance).toBe(1); // Clamped to 1
      expect(normalized.mouseEntropy).toBe(1); // Clamped
      expect(normalized.touchPressureStd).toBe(1); // Clamped
      expect(normalized.deviceStability).toBe(1); // Clamped from 2 to 1
      expect(normalized.sessionDuration).toBe(1); // Normalized from extreme value
      expect(normalized.locationVelocity).toBe(1); // Clamped extreme velocity
    });
  });

  describe('Profile Updates and Exponential Moving Average', () => {
    it('should update user profile with exponential moving average', () => {
      const signal = mockSignals[0];
      const normalized = engine['normalizeSignal'](signal);
      
      engine['updateProfile'](normalized);
      
      const profile = engine['userProfile'].get(sessionId);
      expect(profile).toBeDefined();
      expect(profile!.keystrokeVariance).toBeCloseTo(normalized.keystrokeVariance, 2);
      expect(profile!.mouseEntropy).toBeCloseTo(normalized.mouseEntropy, 2);
      
      // Second update should apply EMA
      const secondSignal = mockSignals[1];
      const normalized2 = engine['normalizeSignal'](secondSignal);
      engine['updateProfile'](normalized2);
      
      const updatedProfile = engine['userProfile'].get(sessionId);
      expect(updatedProfile).toBeDefined();
      
      // EMA should be between first and second values (alpha = 0.3 by default)
      const alpha = 0.3;
      expect(updatedProfile!.keystrokeVariance).toBeCloseTo(
        alpha * normalized2.keystrokeVariance + (1 - alpha) * normalized.keystrokeVariance,
        2
      );
    });

    it('should maintain separate profiles for different sessions', () => {
      const engine2 = new AdvancedRiskEngine('session_2');
      const signal = mockSignals[0];
      const normalized = engine['normalizeSignal'](signal);
      
      engine['updateProfile'](normalized);
      engine2['updateProfile']({ ...normalized, keystrokeVariance: 0.5 });
      
      const profile1 = engine['userProfile'].get(sessionId);
      const profile2 = engine2['userProfile'].get('session_2');
      
      expect(profile1).toBeDefined();
      expect(profile2).toBeDefined();
      expect(profile1!.keystrokeVariance).not.toBeCloseTo(profile2!.keystrokeVariance, 2);
    });

    it('should handle profile updates with missing or null values', () => {
      const partialSignal: BehavioralSignal = {
        keystrokeVariance: 0.2,
        mouseEntropy: null as any, // Invalid
        touchPressureStd: 0.1,
        deviceStability: 0.9,
        sessionDuration: 1200,
        locationVelocity: 30
      };
      
      const normalized = engine['normalizeSignal'](partialSignal);
      engine['updateProfile'](normalized);
      
      const profile = engine['userProfile'].get(sessionId);
      expect(profile).toBeDefined();
      expect(profile!.mouseEntropy).toBe(0); // Should default to 0 for null
    });
  });

  describe('Anomaly Detection', () => {
    it('should detect anomalies using Z-score calculation', () => {
      // Ingest normal signals to establish baseline
      mockSignals.forEach(signal => {
        const normalized = engine['normalizeSignal'](signal);
        engine['sessionHistory'].push(normalized);
      });
      
      // Test anomaly detection on anomalous signal
      const detected = engine['detectAnomalies'](anomalousSignal);
      
      expect(detected).toBe(true); // Should detect high anomaly
      
      // Verify Z-scores for anomalous signal
      const zScores = engine['calculateZScores'](anomalousSignal, engine['sessionHistory']);
      expect(zScores.keystrokeVariance).toBeGreaterThan(2.0); // High variance anomaly
      expect(zScores.mouseEntropy).toBeGreaterThan(2.0); // Low entropy anomaly
      expect(zScores.deviceStability).toBeGreaterThan(2.0); // Low stability
      expect(zScores.locationVelocity).toBeGreaterThan(2.0); // Impossible speed
    });

    it('should not flag normal signals as anomalies', () => {
      mockSignals.forEach(signal => {
        const normalized = engine['normalizeSignal'](signal);
        engine['sessionHistory'].push(normalized);
      });
      
      // Test with another normal signal
      const result = engine['detectAnomalies'](normalSignal);
      expect(result).toBe(false); // Should not detect as anomaly
    });

    it('should handle anomaly detection with insufficient baseline data', () => {
      // Empty history
      const result1 = engine['detectAnomalies'](anomalousSignal);
      expect(result1).toBe(false); // Not enough data to detect anomaly
      
      // Single data point
      engine['sessionHistory'].push(engine['normalizeSignal'](mockSignals[0]));
      const result2 = engine['detectAnomalies'](anomalousSignal);
      expect(result2).toBe(false); // Still insufficient data
    });

    it('should calculate Z-scores correctly for various distributions', () => {
      // Test with mock baseline data
      const baseline = [
        { keystrokeVariance: 0.1 },
        { keystrokeVariance: 0.15 },
        { keystrokeVariance: 0.12 },
        { keystrokeVariance: 0.18 },
        { keystrokeVariance: 0.09 }
      ];
      
      const testSignal = { keystrokeVariance: 0.5 };
      const zScore = engine['calculateZScore'](testSignal.keystrokeVariance, baseline.map(b => b.keystrokeVariance));
      
      // Expected: mean ≈ 0.128, std ≈ 0.036, z-score ≈ (0.5 - 0.128) / 0.036 ≈ 10.2
      expect(zScore).toBeGreaterThan(9);
      expect(zScore).toBeLessThan(11);
    });
  });

  describe('Risk Score Calculation', () => {
    it('should calculate composite risk score with correct weighting', () => {
      // Setup with baseline data
      mockSignals.forEach(signal => engine.ingestSignal(signal));
      
      // Test risk factors for anomalous signal
      const riskFactors: RiskFactors = {
        behavioralDeviation: 0.8,
        deviceAnomaly: 0.7,
        contextualRisk: 0.6,
        temporalPatterns: 0.4
      };
      
      const riskScore = engine['computeCompositeRisk'](riskFactors);
      
      // Expected: 0.4*0.8 + 0.25*0.7 + 0.2*0.6 + 0.15*0.4 = 0.32 + 0.175 + 0.12 + 0.06 = 0.675
      expect(riskScore).toBeCloseTo(0.675, 2);
      expect(riskScore).toBeGreaterThan(0.65); // Medium risk threshold
      expect(riskScore).toBeLessThan(0.7);
    });

    it('should classify risk levels correctly based on composite score', () => {
      const lowRisk: RiskFactors = {
        behavioralDeviation: 0.1,
        deviceAnomaly: 0.05,
        contextualRisk: 0.1,
        temporalPatterns: 0.08
      };
      
      const mediumRisk: RiskFactors = {
        behavioralDeviation: 0.4,
        deviceAnomaly: 0.3,
        contextualRisk: 0.35,
        temporalPatterns: 0.25
      };
      
      const highRisk: RiskFactors = {
        behavioralDeviation: 0.9,
        deviceAnomaly: 0.85,
        contextualRisk: 0.8,
        temporalPatterns: 0.7
      };
      
      expect(engine['getRiskLevel'](engine['computeCompositeRisk'](lowRisk))).toBe('low');
      expect(engine['getRiskLevel'](engine['computeCompositeRisk'](mediumRisk))).toBe('medium');
      expect(engine['getRiskLevel'](engine['computeCompositeRisk'](highRisk))).toBe('high');
    });

    it('should handle edge cases in risk calculation (zero values)', () => {
      const zeroRisk: RiskFactors = {
        behavioralDeviation: 0,
        deviceAnomaly: 0,
        contextualRisk: 0,
        temporalPatterns: 0
      };
      
      const riskScore = engine['computeCompositeRisk'](zeroRisk);
      expect(riskScore).toBe(0);
      expect(engine['getRiskLevel'](riskScore)).toBe('low');
    });

    it('should handle maximum risk values correctly', () => {
      const maxRisk: RiskFactors = {
        behavioralDeviation: 1,
        deviceAnomaly: 1,
        contextualRisk: 1,
        temporalPatterns: 1
      };
      
      const riskScore = engine['computeCompositeRisk'](maxRisk);
      expect(riskScore).toBe(1); // Sum of weights = 1.0
      expect(engine['getRiskLevel'](riskScore)).toBe('high');
    });
  });

  describe('Device Anomaly Detection', () => {
    it('should detect device anomalies based on stability and change frequency', () => {
      // Normal device pattern
      const normalDevice = {
        deviceStability: 0.95,
        deviceChanges: 1, // Single device
        recentDeviceAge: 3600 * 24 * 30 // 30 days old
      };
      
      // Anomalous device pattern
      const anomalousDevice = {
        deviceStability: 0.3,
        deviceChanges: 5, // Multiple device changes
        recentDeviceAge: 300 // 5 minutes old
      };
      
      const normalScore = engine['calculateDeviceAnomaly'](normalDevice);
      const anomalousScore = engine['calculateDeviceAnomaly'](anomalousDevice);
      
      expect(normalScore).toBeLessThan(0.2); // Low anomaly
      expect(anomalousScore).toBeGreaterThan(0.7); // High anomaly
    });

    it('should consider device age in anomaly scoring', () => {
      const newDevice = {
        deviceStability: 0.9,
        deviceChanges: 1,
        recentDeviceAge: 600 // 10 minutes old
      };
      
      const oldDevice = {
        deviceStability: 0.9,
        deviceChanges: 1,
        recentDeviceAge: 86400 * 365 // 1 year old
      };
      
      const newScore = engine['calculateDeviceAnomaly']({ ...newDevice, deviceId: 'new' });
      const oldScore = engine['calculateDeviceAnomaly']({ ...oldDevice, deviceId: 'old' });
      
      expect(newScore).toBeGreaterThan(oldScore); // New devices slightly riskier
      expect(newScore).toBeLessThan(0.4); // Still reasonable
    });

    it('should handle missing device information gracefully', () => {
      const incompleteDevice = {
        deviceStability: 0.8,
        deviceChanges: undefined,
        recentDeviceAge: undefined
      };
      
      const score = engine['calculateDeviceAnomaly'](incompleteDevice);
      expect(score).toBeGreaterThan(0.1); // Some risk for unknown
      expect(score).toBeLessThan(0.5); // Not extreme
    });
  });

  describe('Contextual Risk Assessment', () => {
    it('should calculate contextual risk based on IP reputation and geolocation', () => {
      // Low risk context (good IP, local location)
      const lowContext = {
        ipReputation: 0.95,
        geoDistance: 10, // Local
        timeZoneMatch: true,
        merchantRisk: 0.1
      };
      
      // High risk context (bad IP, distant location)
      const highContext = {
        ipReputation: 0.2,
        geoDistance: 5000, // International
        timeZoneMatch: false,
        merchantRisk: 0.8
      };
      
      const lowScore = engine['calculateContextualRisk'](lowContext);
      const highScore = engine['calculateContextualRisk'](highContext);
      
      expect(lowScore).toBeLessThan(0.2);
      expect(highScore).toBeGreaterThan(0.7);
    });

    it('should penalize timezone mismatches', () => {
      const sameZone = {
        ipReputation: 0.8,
        geoDistance: 50,
        timeZoneMatch: true,
        merchantRisk: 0.3
      };
      
      const differentZone = {
        ...sameZone,
        timeZoneMatch: false
      };
      
      const sameScore = engine['calculateContextualRisk'](sameZone);
      const diffScore = engine['calculateContextualRisk'](differentZone);
      
      expect(diffScore).toBeGreaterThan(sameScore);
      expect(diffScore - sameScore).toBeGreaterThan(0.1); // Significant penalty
    });

    it('should handle missing contextual data', () => {
      const minimalContext = {
        ipReputation: undefined,
        geoDistance: undefined,
        timeZoneMatch: undefined,
        merchantRisk: 0.5
      };
      
      const score = engine['calculateContextualRisk'](minimalContext);
      expect(score).toBeGreaterThan(0.3); // Neutral risk for unknown data
      expect(score).toBeLessThan(0.7);
    });
  });

  describe('Temporal Pattern Analysis', () => {
    it('should detect temporal anomalies (unusual timing patterns)', () => {
      // Normal business hours
      const normalTime = {
        hourOfDay: 14, // 2 PM
        dayOfWeek: 1, // Monday
        timeSinceLast: 3600, // 1 hour since last transaction
        sessionDuration: 1800, // 30 minutes
        transactionVelocity: 2 // 2 transactions per hour
      };
      
      // Suspicious timing (late night, weekend)
      const suspiciousTime = {
        hourOfDay: 2, // 2 AM
        dayOfWeek: 0, // Sunday
        timeSinceLast: 60, // 1 minute since last
        sessionDuration: 120, // 2 minutes
        transactionVelocity: 15 // 15 transactions per hour
      };
      
      const normalScore = engine['analyzeTemporalPatterns'](normalTime);
      const suspiciousScore = engine['analyzeTemporalPatterns'](suspiciousTime);
      
      expect(normalScore).toBeLessThan(0.2);
      expect(suspiciousScore).toBeGreaterThan(0.6);
    });

    it('should identify rapid transaction patterns as high risk', () => {
      const rapidPattern = {
        hourOfDay: 12,
        dayOfWeek: 3, // Wednesday
        timeSinceLast: 30, // 30 seconds since last
        sessionDuration: 300, // 5 minutes
        transactionVelocity: 120 // 2 per minute
      };
      
      const score = engine['analyzeTemporalPatterns'](rapidPattern);
      expect(score).toBeGreaterThan(0.7); // High risk for rapid fire
    });

    it('should handle weekend high-value patterns (potential laundering)', () => {
      const weekendLaundering = {
        hourOfDay: 10,
        dayOfWeek: 6, // Saturday
        timeSinceLast: 7200, // 2 hours
        sessionDuration: 3600, // 1 hour
        transactionVelocity: 1,
        transactionAmount: 25000 // High value
      };
      
      // Add amount to temporal analysis (extension)
      const score = engine['analyzeTemporalPatterns']({ ...weekendLaundering, amount: 25000 });
      expect(score).toBeGreaterThan(0.4); // Elevated risk
    });

    it('should detect seasonal anomalies (off-season high activity)', () => {
      const offSeasonSpike = {
        month: 3, // April (off-season)
        transactionVelocity: 50,
        expectedSeasonalVelocity: 10
      };
      
      const seasonalScore = engine['detectSeasonalAnomaly'](offSeasonSpike);
      expect(seasonalScore).toBeGreaterThan(0.5); // Significant seasonal deviation
    });
  });

  describe('Risk Report Generation', () => {
    it('should generate comprehensive explainable risk reports', () => {
      // Setup baseline
      mockSignals.forEach(signal => engine.ingestSignal(signal));
      
      // Generate report for anomalous case
      const report = engine.generateRiskReport(anomalousSignal);
      
      expect(report).toBeDefined();
      expect(report.riskLevel).toBe('high');
      expect(report.riskScore).toBeGreaterThan(0.7);
      expect(report.explanation).toContain('anomalous');
      expect(report.featureContributions).toBeDefined();
      expect(Array.isArray(report.recommendedActions)).toBe(true);
      expect(report.confidence).toBeGreaterThan(0.8);
    });

    it('should include feature contributions in risk reports', () => {
      mockSignals.forEach(signal => engine.ingestSignal(signal));
      
      const report = engine.generateRiskReport(normalSignal);
      
      const contributions = report.featureContributions;
      expect(contributions).toBeDefined();
      expect(contributions.behavioral).toBeGreaterThan(0);
      expect(contributions.device).toBeGreaterThan(0);
      expect(contributions.contextual).toBeGreaterThan(0);
      expect(contributions.temporal).toBeGreaterThan(0);
      
      // Contributions should sum approximately to 1
      const total = Object.values(contributions).reduce((sum, val) => sum + val, 0);
      expect(total).toBeCloseTo(1.0, 1);
    });

    it('should generate appropriate recommendations based on risk factors', () => {
      const highRiskSignal = anomalousSignal;
      const report = engine.generateRiskReport(highRiskSignal);
      
      const actions = report.recommendedActions;
      expect(actions).toContain('multi-factor authentication');
      expect(actions).toContain('device verification');
      expect(actions).toContain('velocity limits');
      expect(actions.length).toBeGreaterThan(3);
    });

    it('should handle report generation with minimal data', () => {
      const minimalReport = engine.generateRiskReport(normalSignal);
      
      expect(minimalReport).toBeDefined();
      expect(minimalReport.riskLevel).toBe('low');
      expect(minimalReport.riskScore).toBeLessThan(0.4);
      expect(minimalReport.explanation).toContain('normal');
    });
  });

  describe('Integration Testing (Full Flow)', () => {
    it('should process complete fraud detection flow correctly', async () => {
      // Step 1: Ingest normal baseline
      mockSignals.slice(0, 2).forEach(signal => engine.ingestSignal(signal));
      
      // Step 2: Process anomalous transaction
      const startTime = Date.now();
      const result = await engine.analyzeTransaction(anomalousSignal);
      
      // Verify processing
      expect(Date.now() - startTime).toBeLessThan(100); // Should be fast
      expect(engine['sessionHistory'].length).toBe(3); // 2 baseline + 1 anomalous
      
      // Verify risk assessment
      expect(result).toBeDefined();
      expect(result.riskScore).toBeGreaterThan(0.7);
      expect(result.riskLevel).toBe('high');
      expect(result.anomalyScore).toBeGreaterThan(2.0);
      
      // Verify profile update
      const profile = engine['userProfile'].get(sessionId);
      expect(profile).toBeDefined();
      expect(profile!.keystrokeVariance).toBeGreaterThan(0.1); // Updated with anomalous data
    });

    it('should handle concurrent signal processing', async () => {
      // Simulate concurrent transaction processing
      const signals = [mockSignals[0], anomalousSignal, mockSignals[2]];
      const promises = signals.map(signal => engine.analyzeTransaction(signal));
      
      const results = await Promise.all(promises);
      
      expect(results.length).toBe(3);
      expect(results[1].riskScore).toBeGreaterThan(results[0].riskScore); // Anomalous should be higher
      expect(engine['sessionHistory'].length).toBe(3); // All processed
    });

    it('should maintain session isolation in multi-session environment', async () => {
      const engine2 = new AdvancedRiskEngine('session_2');
      
      // Process in parallel across engines
      const results1 = await engine.analyzeTransaction(anomalousSignal);
      const results2 = await engine2.analyzeTransaction(normalSignal);
      
      // Verify isolation
      expect(results1.riskScore).toBeGreaterThan(0.7);
      expect(results2.riskScore).toBeLessThan(0.4);
      
      // Verify separate histories
      expect(engine['sessionHistory'].length).toBe(1);
      expect(engine2['sessionHistory'].length).toBe(1);
      expect(engine['userProfile'].size).toBe(1);
      expect(engine2['userProfile'].size).toBe(1);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle invalid signal data gracefully', () => {
      const invalidSignal = {
        keystrokeVariance: NaN as any,
        mouseEntropy: Infinity as any,
        touchPressureStd: -Infinity as any,
        deviceStability: 'invalid' as any,
        sessionDuration: -100,
        locationVelocity: NaN as any
      };
      
      // Should not throw
      expect(() => engine.ingestSignal(invalidSignal)).not.toThrow();
      
      // Should normalize to safe values
      const normalized = engine['normalizeSignal'](invalidSignal);
      expect(normalized.keystrokeVariance).toBe(0);
      expect(isFinite(normalized.mouseEntropy)).toBe(true);
      expect(normalized.touchPressureStd).toBe(0);
      expect(normalized.deviceStability).toBe(0);
      expect(normalized.sessionDuration).toBe(0);
      expect(normalized.locationVelocity).toBe(0);
    });

    it('should handle empty session history without crashing', () => {
      // Empty session
      const emptyResult = engine.generateRiskReport(normalSignal);
      expect(emptyResult).toBeDefined();
      expect(emptyResult.riskLevel).toBe('unknown'); // Or default low
      expect(emptyResult.anomalyScore).toBe(0);
    });

    it('should validate risk factor bounds (0-1)', () => {
      const invalidFactors: RiskFactors = {
        behavioralDeviation: 1.5, // Invalid >1
        deviceAnomaly: -0.2, // Invalid <0
        contextualRisk: NaN as any,
        temporalPatterns: Infinity as any
      };
      
      const validated = engine['validateRiskFactors'](invalidFactors);
      
      expect(validated.behavioralDeviation).toBe(1); // Clamped
      expect(validated.deviceAnomaly).toBe(0); // Clamped
      expect(validated.contextualRisk).toBe(0); // NaN to 0
      expect(validated.temporalPatterns).toBe(1); // Infinity to 1
    });
  });

  describe('Performance Testing', () => {
    it('should process 1000 signals efficiently (<500ms)', async () => {
      const signals = Array.from({ length: 1000 }, (_, i) => ({
        keystrokeVariance: Math.random() * 0.3,
        mouseEntropy: 1.5 + Math.random() * 1,
        touchPressureStd: Math.random() * 0.1,
        deviceStability: 0.8 + Math.random() * 0.2,
        sessionDuration: 600 + Math.random() * 2400,
        locationVelocity: Math.random() * 100
      }));
      
      const startTime = performance.now();
      const results = await Promise.all(
        signals.map(signal => engine.analyzeTransaction(signal))
      );
      const endTime = performance.now();
      
      const processingTime = endTime - startTime;
      expect(processingTime).toBeLessThan(500); // <500ms for 1000 signals
      expect(results.length).toBe(1000);
      expect(engine['sessionHistory'].length).toBe(1000);
    });

    it('should maintain memory efficiency with large session history', () => {
      // Fill history with 10,000 signals
      const signals = Array.from({ length: 10000 }, () => mockSignals[0]);
      
      signals.forEach(signal => engine.ingestSignal(signal));
      
      expect(engine['sessionHistory'].length).toBe(10000);
      
      // Memory should be reasonable (rough estimate)
      const memoryUsage = performance.memory?.usedJSHeapSize || 0;
      expect(memoryUsage).toBeLessThan(100 * 1024 * 1024); // <100MB (approximate)
    });
  });
});
