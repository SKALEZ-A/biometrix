import { SanitizationUtils } from '../../src/utils/sanitization.utils';

describe('SanitizationUtils', () => {
  describe('sanitizeHTML', () => {
    it('should escape HTML special characters', () => {
      const input = '<script>alert("xss")</script>';
      const result = SanitizationUtils.sanitizeHTML(input);
      expect(result).not.toContain('<script>');
      expect(result).toContain('&lt;script&gt;');
    });

    it('should escape quotes', () => {
      const input = 'Hello "World" and \'Universe\'';
      const result = SanitizationUtils.sanitizeHTML(input);
      expect(result).toContain('&quot;');
      expect(result).toContain('&#x27;');
    });
  });

  describe('sanitizeSQL', () => {
    it('should escape single quotes', () => {
      const input = "O'Reilly";
      const result = SanitizationUtils.sanitizeSQL(input);
      expect(result).toBe("O''Reilly");
    });

    it('should remove SQL injection attempts', () => {
      const input = "'; DROP TABLE users; --";
      const result = SanitizationUtils.sanitizeSQL(input);
      expect(result).not.toContain(';');
      expect(result).not.toContain('--');
    });
  });

  describe('sanitizeFilename', () => {
    it('should remove invalid characters', () => {
      const input = 'my<file>name?.txt';
      const result = SanitizationUtils.sanitizeFilename(input);
      expect(result).toBe('my_file_name_.txt');
    });

    it('should prevent directory traversal', () => {
      const input = '../../../etc/passwd';
      const result = SanitizationUtils.sanitizeFilename(input);
      expect(result).not.toContain('..');
    });
  });

  describe('removeXSS', () => {
    it('should remove script tags', () => {
      const input = 'Hello <script>alert("xss")</script> World';
      const result = SanitizationUtils.removeXSS(input);
      expect(result).not.toContain('<script>');
      expect(result).toContain('Hello');
      expect(result).toContain('World');
    });

    it('should remove event handlers', () => {
      const input = '<div onclick="alert(\'xss\')">Click me</div>';
      const result = SanitizationUtils.removeXSS(input);
      expect(result).not.toContain('onclick');
    });
  });

  describe('stripTags', () => {
    it('should remove all HTML tags', () => {
      const input = '<p>Hello <strong>World</strong></p>';
      const result = SanitizationUtils.stripTags(input);
      expect(result).toBe('Hello World');
    });
  });

  describe('truncate', () => {
    it('should truncate long text', () => {
      const input = 'This is a very long text that needs to be truncated';
      const result = SanitizationUtils.truncate(input, 20);
      expect(result.length).toBeLessThanOrEqual(20);
      expect(result).toContain('...');
    });

    it('should not truncate short text', () => {
      const input = 'Short text';
      const result = SanitizationUtils.truncate(input, 20);
      expect(result).toBe(input);
    });
  });
});
