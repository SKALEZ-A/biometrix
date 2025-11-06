import { describe, it, expect } from '@jest/globals';
import {
  sha256,
  sha512,
  md5,
  hmacSha256,
  hashPassword,
  verifyPassword,
  generateSalt,
  generateRandomToken,
  hashObject
} from '../../src/utils/hash.utils';

describe('Hash Utils', () => {
  describe('sha256', () => {
    it('should generate consistent SHA256 hash', () => {
      const data = 'test data';
      const hash1 = sha256(data);
      const hash2 = sha256(data);

      expect(hash1).toBe(hash2);
      expect(hash1).toHaveLength(64);
    });

    it('should generate different hashes for different data', () => {
      const hash1 = sha256('data1');
      const hash2 = sha256('data2');

      expect(hash1).not.toBe(hash2);
    });
  });

  describe('hmacSha256', () => {
    it('should generate HMAC with secret', () => {
      const data = 'test data';
      const secret = 'secret-key';
      const hmac = hmacSha256(data, secret);

      expect(hmac).toHaveLength(64);
    });

    it('should generate different HMACs with different secrets', () => {
      const data = 'test data';
      const hmac1 = hmacSha256(data, 'secret1');
      const hmac2 = hmacSha256(data, 'secret2');

      expect(hmac1).not.toBe(hmac2);
    });
  });

  describe('password hashing', () => {
    it('should hash and verify password correctly', () => {
      const password = 'mySecurePassword123';
      const salt = generateSalt();
      const hash = hashPassword(password, salt);

      expect(verifyPassword(password, salt, hash)).toBe(true);
      expect(verifyPassword('wrongPassword', salt, hash)).toBe(false);
    });

    it('should generate unique salts', () => {
      const salt1 = generateSalt();
      const salt2 = generateSalt();

      expect(salt1).not.toBe(salt2);
    });
  });

  describe('generateRandomToken', () => {
    it('should generate random tokens', () => {
      const token1 = generateRandomToken();
      const token2 = generateRandomToken();

      expect(token1).not.toBe(token2);
      expect(token1.length).toBeGreaterThan(0);
    });

    it('should generate tokens of specified length', () => {
      const token = generateRandomToken(16);
      const decoded = Buffer.from(token, 'base64url');

      expect(decoded.length).toBe(16);
    });
  });

  describe('hashObject', () => {
    it('should generate consistent hash for same object', () => {
      const obj = { name: 'John', age: 30, city: 'NYC' };
      const hash1 = hashObject(obj);
      const hash2 = hashObject(obj);

      expect(hash1).toBe(hash2);
    });

    it('should generate same hash regardless of property order', () => {
      const obj1 = { name: 'John', age: 30 };
      const obj2 = { age: 30, name: 'John' };

      expect(hashObject(obj1)).toBe(hashObject(obj2));
    });
  });
});
