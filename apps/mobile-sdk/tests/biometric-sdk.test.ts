import { BiometricSDK } from '../src/biometric-sdk';

describe('BiometricSDK', () => {
  let sdk: BiometricSDK;

  beforeEach(() => {
    sdk = new BiometricSDK({ apiKey: 'test-key', endpoint: 'http://localhost:3000' });
  });

  describe('initialization', () => {
    it('should initialize with config', () => {
      expect(sdk).toBeDefined();
    });

    it('should throw error without API key', () => {
      expect(() => new BiometricSDK({ apiKey: '', endpoint: '' })).toThrow();
    });
  });

  describe('capture', () => {
    it('should capture biometric data', async () => {
      const result = await sdk.capture('fingerprint');
      expect(result).toHaveProperty('data');
      expect(result).toHaveProperty('quality');
    });

    it('should validate biometric type', async () => {
      await expect(sdk.capture('invalid' as any)).rejects.toThrow();
    });
  });

  describe('verify', () => {
    it('should verify biometric data', async () => {
      const captureResult = await sdk.capture('fingerprint');
      const verifyResult = await sdk.verify('user123', captureResult.data);
      expect(verifyResult).toHaveProperty('verified');
      expect(verifyResult).toHaveProperty('confidence');
    });

    it('should return false for invalid data', async () => {
      const result = await sdk.verify('user123', 'invalid-data');
      expect(result.verified).toBe(false);
    });
  });

  describe('enroll', () => {
    it('should enroll new biometric', async () => {
      const captureResult = await sdk.capture('fingerprint');
      const enrollResult = await sdk.enroll('user123', captureResult.data);
      expect(enrollResult).toHaveProperty('success');
      expect(enrollResult.success).toBe(true);
    });
  });
});
