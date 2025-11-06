import { Request, Response, NextFunction } from 'express';
import validator from 'validator';

export class RequestSanitizerMiddleware {
  public static sanitize(req: Request, res: Response, next: NextFunction): void {
    if (req.body) {
      req.body = RequestSanitizerMiddleware.sanitizeObject(req.body);
    }

    if (req.query) {
      req.query = RequestSanitizerMiddleware.sanitizeObject(req.query);
    }

    if (req.params) {
      req.params = RequestSanitizerMiddleware.sanitizeObject(req.params);
    }

    next();
  }

  private static sanitizeObject(obj: any): any {
    if (typeof obj === 'string') {
      return RequestSanitizerMiddleware.sanitizeString(obj);
    }

    if (Array.isArray(obj)) {
      return obj.map(item => RequestSanitizerMiddleware.sanitizeObject(item));
    }

    if (obj !== null && typeof obj === 'object') {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(obj)) {
        sanitized[key] = RequestSanitizerMiddleware.sanitizeObject(value);
      }
      return sanitized;
    }

    return obj;
  }

  private static sanitizeString(str: string): string {
    let sanitized = str.trim();
    sanitized = validator.escape(sanitized);
    sanitized = sanitized.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    sanitized = sanitized.replace(/javascript:/gi, '');
    sanitized = sanitized.replace(/on\w+\s*=/gi, '');
    return sanitized;
  }

  public static sanitizeSQL(req: Request, res: Response, next: NextFunction): void {
    const sqlInjectionPatterns = [
      /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)/gi,
      /(--|\;|\/\*|\*\/|xp_|sp_)/gi,
      /(\bOR\b.*=.*|1=1|'=')/gi,
    ];

    const checkForSQLInjection = (value: any): boolean => {
      if (typeof value === 'string') {
        return sqlInjectionPatterns.some(pattern => pattern.test(value));
      }
      return false;
    };

    const scanObject = (obj: any): boolean => {
      if (typeof obj === 'string') {
        return checkForSQLInjection(obj);
      }
      if (Array.isArray(obj)) {
        return obj.some(item => scanObject(item));
      }
      if (obj !== null && typeof obj === 'object') {
        return Object.values(obj).some(value => scanObject(value));
      }
      return false;
    };

    if (scanObject(req.body) || scanObject(req.query) || scanObject(req.params)) {
      return res.status(400).json({ error: 'Potential SQL injection detected' });
    }

    next();
  }
}
