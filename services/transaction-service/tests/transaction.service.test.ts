import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { TransactionService } from '../src/services/transaction.service';

describe('TransactionService', () => {
  let service: TransactionService;

  beforeEach(() => {
    service = new TransactionService();
  });

  describe('createTransaction', () => {
    it('should create a transaction successfully', async () => {
      const transactionData = {
        userId: 'user-123',
        merchantId: 'merchant-456',
        amount: 100.50,
        currency: 'USD',
        type: 'purchase' as const,
        paymentMethod: 'card' as const,
        metadata: {
          deviceFingerprint: 'device-123',
          ipAddress: '192.168.1.1',
        },
      };

      const result = await service.createTransaction(transactionData);

      expect(result).toHaveProperty('id');
      expect(result.userId).toBe(transactionData.userId);
      expect(result.amount).toBe(transactionData.amount);
      expect(result.status).toBe('pending');
    });

    it('should reject transaction with invalid amount', async () => {
      const transactionData = {
        userId: 'user-123',
        merchantId: 'merchant-456',
        amount: -100,
        currency: 'USD',
        type: 'purchase' as const,
        paymentMethod: 'card' as const,
        metadata: {
          deviceFingerprint: 'device-123',
          ipAddress: '192.168.1.1',
        },
      };

      await expect(service.createTransaction(transactionData)).rejects.toThrow();
    });
  });

  describe('getTransaction', () => {
    it('should retrieve transaction by id', async () => {
      const transactionId = 'txn-123';
      const transaction = await service.getTransaction(transactionId);

      expect(transaction).toHaveProperty('id', transactionId);
    });

    it('should return null for non-existent transaction', async () => {
      const transaction = await service.getTransaction('non-existent');
      expect(transaction).toBeNull();
    });
  });

  describe('calculateRiskScore', () => {
    it('should calculate risk score for transaction', async () => {
      const transactionData = {
        userId: 'user-123',
        amount: 1000,
        merchantId: 'merchant-456',
        location: {
          latitude: 40.7128,
          longitude: -74.0060,
          country: 'US',
        },
      };

      const riskScore = await service.calculateRiskScore(transactionData);

      expect(riskScore).toBeGreaterThanOrEqual(0);
      expect(riskScore).toBeLessThanOrEqual(100);
    });
  });
});
