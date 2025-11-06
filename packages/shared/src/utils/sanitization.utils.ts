export class SanitizationUtils {
  static sanitizeHTML(input: string): string {
    return input
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;')
      .replace(/\//g, '&#x2F;');
  }

  static sanitizeSQL(input: string): string {
    return input
      .replace(/'/g, "''")
      .replace(/;/g, '')
      .replace(/--/g, '')
      .replace(/\/\*/g, '')
      .replace(/\*\//g, '');
  }

  static sanitizeFilename(filename: string): string {
    return filename
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .replace(/\.{2,}/g, '.')
      .substring(0, 255);
  }

  static sanitizeEmail(email: string): string {
    return email.toLowerCase().trim();
  }

  static sanitizePhoneNumber(phone: string): string {
    return phone.replace(/[^0-9+]/g, '');
  }

  static sanitizeURL(url: string): string {
    try {
      const parsed = new URL(url);
      return parsed.toString();
    } catch {
      return '';
    }
  }

  static removeXSS(input: string): string {
    const patterns = [
      /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
      /javascript:/gi,
      /on\w+\s*=/gi,
      /<iframe/gi,
      /<object/gi,
      /<embed/gi
    ];

    let sanitized = input;
    patterns.forEach(pattern => {
      sanitized = sanitized.replace(pattern, '');
    });

    return sanitized;
  }

  static sanitizeJSON(obj: any): any {
    if (typeof obj === 'string') {
      return this.sanitizeHTML(obj);
    }

    if (Array.isArray(obj)) {
      return obj.map(item => this.sanitizeJSON(item));
    }

    if (obj && typeof obj === 'object') {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(obj)) {
        sanitized[key] = this.sanitizeJSON(value);
      }
      return sanitized;
    }

    return obj;
  }

  static stripTags(html: string): string {
    return html.replace(/<[^>]*>/g, '');
  }

  static truncate(text: string, maxLength: number, suffix: string = '...'): string {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - suffix.length) + suffix;
  }
}
