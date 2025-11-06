import { MerchantRiskScoringService } from '../src/services/merchant-risk-scoring.service';

describe('MerchantRiskScoringService', () => {
  let service: MerchantRiskScoringService;

  beforeEach(() => {
    service = new MerchantRiskScoringService();
  });

  describe('calculateRiskScore', () => {
    it('should calculate risk score for merchant', async () => {
      const merchantData = {
        merchantId: 'merchant123',
        transactionVolume: 100000,
        chargebackRate: 0.01,
        averageTicketSize: 50,
        businessAge: 365
      };

      const score = await service.calculateRiskScore(merchantData);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it('should return high risk for high chargeback rate', async () => {
      const merchantData = {
        merchantId: 'merchant123',
        transactionVolume: 100000,
        chargebackRate: 0.05,
        averageTicketSize: 50,
        businessAge: 365
      };

      const score = await service.calculateRiskScore(merchantData);
      expect(score).toBeGreaterThan(0.7);
    });

    it('should return low risk for established merchant', async () => {
      const merchantData = {
        merchantId: 'merchant123',
        transactionVolume: 1000000,
        chargebackRate: 0.001,
        averageTicketSize: 50,
        businessAge: 1825
      };

      const score = await service.calculateRiskScore(merchantData);
      expect(score).toBeLessThan(0.3);
    });
  });

  describe('analyzeChargebackPattern', () => {
    it('should detect chargeback patterns', async () => {
      const chargebacks = [
        { date: new Date(), amount: 100, reason: 'fraud' },
        { date: new Date(), amount: 150, reason: 'fraud' },
        { date: new Date(), amount: 200, reason: 'fraud' }
      ];

      const pattern = await service.analyzeChargebackPattern('merchant123', chargebacks);
      expect(pattern).toHaveProperty('isAnomalous');
      expect(pattern).toHaveProperty('riskLevel');
    });
  });
});
