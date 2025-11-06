import { describe, it, expect, beforeEach } from '@jest/globals';
import { KYCVerificationService } from '../src/services/kyc-verification.service';

describe('KYCVerificationService', () => {
  let kycService: KYCVerificationService;

  beforeEach(() => {
    kycService = new KYCVerificationService();
  });

  describe('verifyIdentity', () => {
    it('should verify valid identity documents', async () => {
      const documents = {
        passport: {
          number: 'AB123456',
          country: 'US',
          expiryDate: new Date('2030-12-31')
        },
        personalInfo: {
          firstName: 'John',
          lastName: 'Doe',
          dateOfBirth: new Date('1990-01-01'),
          nationality: 'US'
        }
      };

      const result = await kycService.verifyIdentity('user-123', documents);

      expect(result.status).toBe('approved');
      expect(result.verificationLevel).toBe('enhanced');
    });

    it('should reject expired documents', async () => {
      const documents = {
        passport: {
          number: 'AB123456',
          country: 'US',
          expiryDate: new Date('2020-12-31')
        },
        personalInfo: {
          firstName: 'John',
          lastName: 'Doe',
          dateOfBirth: new Date('1990-01-01'),
          nationality: 'US'
        }
      };

      const result = await kycService.verifyIdentity('user-123', documents);

      expect(result.status).toBe('rejected');
      expect(result.rejectionReason).toContain('expired');
    });
  });

  describe('checkDocumentAuthenticity', () => {
    it('should detect forged documents', async () => {
      const document = {
        type: 'passport',
        imageData: 'base64-encoded-image',
        metadata: {
          hasWatermark: false,
          hasHologram: false
        }
      };

      const isAuthentic = await kycService.checkDocumentAuthenticity(document);

      expect(isAuthentic).toBe(false);
    });
  });
});
