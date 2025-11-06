import { NetworkUtils } from '../../src/utils/network.utils';

describe('NetworkUtils', () => {
  describe('isPrivateIP', () => {
    it('should identify private IPs', () => {
      expect(NetworkUtils.isPrivateIP('10.0.0.1')).toBe(true);
      expect(NetworkUtils.isPrivateIP('172.16.0.1')).toBe(true);
      expect(NetworkUtils.isPrivateIP('192.168.1.1')).toBe(true);
      expect(NetworkUtils.isPrivateIP('127.0.0.1')).toBe(true);
    });

    it('should identify public IPs', () => {
      expect(NetworkUtils.isPrivateIP('8.8.8.8')).toBe(false);
      expect(NetworkUtils.isPrivateIP('1.1.1.1')).toBe(false);
    });
  });

  describe('isValidIPv4', () => {
    it('should validate correct IPv4 addresses', () => {
      expect(NetworkUtils.isValidIPv4('192.168.1.1')).toBe(true);
      expect(NetworkUtils.isValidIPv4('8.8.8.8')).toBe(true);
      expect(NetworkUtils.isValidIPv4('0.0.0.0')).toBe(true);
    });

    it('should reject invalid IPv4 addresses', () => {
      expect(NetworkUtils.isValidIPv4('256.1.1.1')).toBe(false);
      expect(NetworkUtils.isValidIPv4('192.168.1')).toBe(false);
      expect(NetworkUtils.isValidIPv4('invalid')).toBe(false);
    });
  });

  describe('anonymizeIP', () => {
    it('should anonymize IPv4 address', () => {
      const anonymized = NetworkUtils.anonymizeIP('192.168.1.100');
      expect(anonymized).toBe('192.168.1.0');
    });

    it('should hash non-IPv4 addresses', () => {
      const anonymized = NetworkUtils.anonymizeIP('invalid-ip');
      expect(anonymized).toHaveLength(16);
    });
  });

  describe('parseUserAgent', () => {
    it('should parse Chrome user agent', () => {
      const ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36';
      const parsed = NetworkUtils.parseUserAgent(ua);
      
      expect(parsed.browser).toBe('Chrome');
      expect(parsed.os).toBe('Windows');
      expect(parsed.device).toBe('Desktop');
    });

    it('should parse Firefox user agent', () => {
      const ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0';
      const parsed = NetworkUtils.parseUserAgent(ua);
      
      expect(parsed.browser).toBe('Firefox');
      expect(parsed.os).toBe('Windows');
    });

    it('should parse mobile user agent', () => {
      const ua = 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1';
      const parsed = NetworkUtils.parseUserAgent(ua);
      
      expect(parsed.device).toBe('Mobile');
      expect(parsed.os).toBe('iOS');
    });
  });
});
