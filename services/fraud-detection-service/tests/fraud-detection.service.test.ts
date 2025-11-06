import { describe, it, expect, beforeEach } from '@jest/globals';
import { RiskEngineService } from '../src/services/risk-engine.service';
import { FeatureEngineeringService } from '../src/services/feature-engineering.service';

describe('RiskEngineService', () => {
  let riskEngine: RiskEngineService;
  let featureEngineering: FeatureEngineeringService;

  beforeEach(() => {
    featureEngineering = new FeatureEngineeringService();
    riskEngine = new RiskEngineService(featureEngineering);
  });

  describe('calculateRiskScore', () => {
    it('should return high risk score for suspicious transaction', async () => {
      const transaction = {
        userId: 'user-123',
        amount: 10000,
        currency: 'USD',
        location: {
          country: 'Unknown',
          city: 'Unknown',
          latitude: 0,
          longitude: 0
        },
        deviceFingerprint: 'new-device',
        ipAddress: '1.2.3.4',
        timestamp: new Date()
      };

      const riskScore = await riskEngine.calculateRiskScore(transaction);

      expect(riskScore).toBeGreaterThan(50);
      expect(riskScore).toBeLessThanOrEqual(100);
    });

    it('should return low risk score for normal transaction', async () => {
      const transaction = {
        userId: 'user-123',
        amount: 50,
        currency: 'USD',
        location: {
          country: 'US',
          city: 'New York',
          latitude: 40.7128,
          longitude: -74.0060
        },
        deviceFingerprint: 'known-device',
        ipAddress: '192.168.1.1',
        timestamp: new Date()
      };

      const riskScore = await riskEngine.calculateRiskScore(transaction);

      expect(riskScore).toBeLessThan(50);
      expect(riskScore).toBeGreaterThanOrEqual(0);
    });
  });

  describe('detectAnomalies', () => {
    it('should detect velocity anomalies', async () => {
      const transactions = Array.from({ length: 10 }, (_, i) => ({
        userId: 'user-123',
        amount: 1000,
        timestamp: new Date(Date.now() - i * 60000)
      }));

      const anomalies = await riskEngine.detectAnomalies(transactions);

      expect(anomalies).toBeDefined();
      expect(anomalies.velocityAnomaly).toBe(true);
    });
  });
});
