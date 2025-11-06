import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { FraudDetectionClient } from '../src/fraud-detection-client';

global.fetch = jest.fn();

describe('FraudDetectionClient', () => {
  let client: FraudDetectionClient;

  beforeEach(() => {
    client = new FraudDetectionClient('https://api.example.com', 'test-api-key');
    (fetch as jest.Mock).mockClear();
  });

  describe('detectFraud', () => {
    it('should send fraud detection request', async () => {
      const mockResponse = {
        transactionId: 'txn-123',
        riskScore: 75,
        fraudProbability: 0.85,
        decision: 'review',
        reasons: ['High amount', 'New location'],
        recommendedActions: ['Manual review required']
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const request = {
        transactionId: 'txn-123',
        userId: 'user-456',
        amount: 5000,
        currency: 'USD',
        merchantId: 'merchant-789',
        deviceFingerprint: 'device-abc'
      };

      const result = await client.detectFraud(request);

      expect(result).toEqual(mockResponse);
      expect(fetch).toHaveBeenCalledWith(
        'https://api.example.com/fraud-detection/detect',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-api-key'
          })
        })
      );
    });

    it('should throw error on API failure', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        statusText: 'Internal Server Error'
      });

      const request = {
        transactionId: 'txn-123',
        userId: 'user-456',
        amount: 5000,
        currency: 'USD',
        merchantId: 'merchant-789',
        deviceFingerprint: 'device-abc'
      };

      await expect(client.detectFraud(request)).rejects.toThrow(
        'Fraud detection failed: Internal Server Error'
      );
    });
  });

  describe('getRiskScore', () => {
    it('should retrieve risk score for transaction', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ riskScore: 65 })
      });

      const score = await client.getRiskScore('txn-123');

      expect(score).toBe(65);
    });
  });
});
