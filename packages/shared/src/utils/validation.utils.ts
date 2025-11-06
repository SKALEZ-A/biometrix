import { logger } from './logger';

export class ValidationUtils {
  public static isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  public static isValidPhoneNumber(phone: string): boolean {
    const phoneRegex = /^\+?[1-9]\d{1,14}$/;
    return phoneRegex.test(phone.replace(/[\s-()]/g, ''));
  }

  public static isValidURL(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  public static isValidIPAddress(ip: string): boolean {
    const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
    const ipv6Regex = /^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$/;
    return ipv4Regex.test(ip) || ipv6Regex.test(ip);
  }

  public static isValidCurrency(currency: string): boolean {
    const validCurrencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'INR'];
    return validCurrencies.includes(currency.toUpperCase());
  }

  public static isValidAmount(amount: number): boolean {
    return typeof amount === 'number' && amount > 0 && isFinite(amount);
  }

  public static isValidCountryCode(code: string): boolean {
    return /^[A-Z]{2}$/.test(code);
  }

  public static sanitizeString(input: string): string {
    return input.replace(/[<>\"'&]/g, (char) => {
      const entities: { [key: string]: string } = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '&': '&amp;',
      };
      return entities[char] || char;
    });
  }

  public static validateTransactionAmount(amount: number, currency: string): {
    valid: boolean;
    error?: string;
  } {
    if (!this.isValidAmount(amount)) {
      return { valid: false, error: 'Invalid amount' };
    }

    if (!this.isValidCurrency(currency)) {
      return { valid: false, error: 'Invalid currency' };
    }

    const limits: { [key: string]: { min: number; max: number } } = {
      USD: { min: 0.01, max: 1000000 },
      EUR: { min: 0.01, max: 1000000 },
      GBP: { min: 0.01, max: 1000000 },
    };

    const limit = limits[currency.toUpperCase()];
    if (limit && (amount < limit.min || amount > limit.max)) {
      return {
        valid: false,
        error: `Amount must be between ${limit.min} and ${limit.max} ${currency}`,
      };
    }

    return { valid: true };
  }

  public static validateBiometricData(data: any): { valid: boolean; error?: string } {
    if (!data || typeof data !== 'object') {
      return { valid: false, error: 'Invalid biometric data format' };
    }

    if (data.type === 'fingerprint' && !data.template) {
      return { valid: false, error: 'Fingerprint template is required' };
    }

    if (data.type === 'facial' && !data.faceEncoding) {
      return { valid: false, error: 'Face encoding is required' };
    }

    if (data.type === 'iris' && !data.irisPattern) {
      return { valid: false, error: 'Iris pattern is required' };
    }

    return { valid: true };
  }

  public static validateCoordinates(lat: number, lon: number): boolean {
    return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
  }

  public static isValidUUID(uuid: string): boolean {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    return uuidRegex.test(uuid);
  }

  public static validatePasswordStrength(password: string): {
    valid: boolean;
    score: number;
    feedback: string[];
  } {
    const feedback: string[] = [];
    let score = 0;

    if (password.length >= 8) score++;
    else feedback.push('Password should be at least 8 characters');

    if (/[a-z]/.test(password)) score++;
    else feedback.push('Include lowercase letters');

    if (/[A-Z]/.test(password)) score++;
    else feedback.push('Include uppercase letters');

    if (/\d/.test(password)) score++;
    else feedback.push('Include numbers');

    if (/[^a-zA-Z0-9]/.test(password)) score++;
    else feedback.push('Include special characters');

    return {
      valid: score >= 4,
      score,
      feedback,
    };
  }
}
