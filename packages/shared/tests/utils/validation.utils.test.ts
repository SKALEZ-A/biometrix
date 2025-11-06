import { describe, it, expect } from '@jest/globals';
import { ValidationUtils } from '../../src/utils/validation.utils';

describe('ValidationUtils', () => {
  describe('isValidEmail', () => {
    it('should validate correct email', () => {
      expect(ValidationUtils.isValidEmail('test@example.com')).toBe(true);
      expect(ValidationUtils.isValidEmail('user.name@domain.co.uk')).toBe(true);
    });

    it('should reject invalid email', () => {
      expect(ValidationUtils.isValidEmail('invalid')).toBe(false);
      expect(ValidationUtils.isValidEmail('test@')).toBe(false);
      expect(ValidationUtils.isValidEmail('@example.com')).toBe(false);
    });
  });

  describe('isValidPhoneNumber', () => {
    it('should validate correct phone number', () => {
      expect(ValidationUtils.isValidPhoneNumber('+1234567890')).toBe(true);
      expect(ValidationUtils.isValidPhoneNumber('+44 20 1234 5678')).toBe(true);
    });

    it('should reject invalid phone number', () => {
      expect(ValidationUtils.isValidPhoneNumber('123')).toBe(false);
      expect(ValidationUtils.isValidPhoneNumber('abc')).toBe(false);
    });
  });

  describe('isValidURL', () => {
    it('should validate correct URL', () => {
      expect(ValidationUtils.isValidURL('https://example.com')).toBe(true);
      expect(ValidationUtils.isValidURL('http://localhost:3000')).toBe(true);
    });

    it('should reject invalid URL', () => {
      expect(ValidationUtils.isValidURL('not-a-url')).toBe(false);
      expect(ValidationUtils.isValidURL('ftp://invalid')).toBe(false);
    });
  });

  describe('isValidIPAddress', () => {
    it('should validate IPv4 address', () => {
      expect(ValidationUtils.isValidIPAddress('192.168.1.1')).toBe(true);
      expect(ValidationUtils.isValidIPAddress('10.0.0.1')).toBe(true);
    });

    it('should reject invalid IP address', () => {
      expect(ValidationUtils.isValidIPAddress('256.1.1.1')).toBe(false);
      expect(ValidationUtils.isValidIPAddress('invalid')).toBe(false);
    });
  });

  describe('validateTransactionAmount', () => {
    it('should validate correct amount', () => {
      const result = ValidationUtils.validateTransactionAmount(100, 'USD');
      expect(result.valid).toBe(true);
    });

    it('should reject negative amount', () => {
      const result = ValidationUtils.validateTransactionAmount(-100, 'USD');
      expect(result.valid).toBe(false);
    });

    it('should reject invalid currency', () => {
      const result = ValidationUtils.validateTransactionAmount(100, 'INVALID');
      expect(result.valid).toBe(false);
    });
  });

  describe('validatePasswordStrength', () => {
    it('should validate strong password', () => {
      const result = ValidationUtils.validatePasswordStrength('StrongP@ss123');
      expect(result.valid).toBe(true);
      expect(result.score).toBeGreaterThanOrEqual(4);
    });

    it('should reject weak password', () => {
      const result = ValidationUtils.validatePasswordStrength('weak');
      expect(result.valid).toBe(false);
      expect(result.feedback.length).toBeGreaterThan(0);
    });
  });

  describe('isValidUUID', () => {
    it('should validate correct UUID', () => {
      expect(ValidationUtils.isValidUUID('123e4567-e89b-12d3-a456-426614174000')).toBe(true);
    });

    it('should reject invalid UUID', () => {
      expect(ValidationUtils.isValidUUID('not-a-uuid')).toBe(false);
      expect(ValidationUtils.isValidUUID('123-456-789')).toBe(false);
    });
  });
});
