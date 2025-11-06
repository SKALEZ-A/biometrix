import { Request, Response, NextFunction } from 'express';

export class SecurityHeadersMiddleware {
  private static readonly SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
  };

  public static apply(req: Request, res: Response, next: NextFunction): void {
    Object.entries(SecurityHeadersMiddleware.SECURITY_HEADERS).forEach(([header, value]) => {
      res.setHeader(header, value);
    });

    res.removeHeader('X-Powered-By');
    next();
  }

  public static applyCustomCSP(cspDirectives: Record<string, string>) {
    return (req: Request, res: Response, next: NextFunction): void => {
      const csp = Object.entries(cspDirectives)
        .map(([key, value]) => `${key} ${value}`)
        .join('; ');
      res.setHeader('Content-Security-Policy', csp);
      next();
    };
  }

  public static enableHSTS(maxAge: number = 31536000, includeSubDomains: boolean = true) {
    return (req: Request, res: Response, next: NextFunction): void => {
      const hstsValue = `max-age=${maxAge}${includeSubDomains ? '; includeSubDomains' : ''}; preload`;
      res.setHeader('Strict-Transport-Security', hstsValue);
      next();
    };
  }
}
