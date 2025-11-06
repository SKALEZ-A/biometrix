import { Request, Response, NextFunction } from 'express';

interface CorsOptions {
  origin?: string | string[] | ((origin: string) => boolean);
  methods?: string[];
  allowedHeaders?: string[];
  exposedHeaders?: string[];
  credentials?: boolean;
  maxAge?: number;
  preflightContinue?: boolean;
  optionsSuccessStatus?: number;
}

export class CorsHandler {
  private options: Required<CorsOptions>;
  private allowedOrigins: Set<string>;

  constructor(options: CorsOptions = {}) {
    this.options = {
      origin: options.origin || '*',
      methods: options.methods || ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
      allowedHeaders: options.allowedHeaders || [
        'Content-Type',
        'Authorization',
        'X-Requested-With',
        'X-API-Key',
        'X-Device-ID',
        'X-Session-ID'
      ],
      exposedHeaders: options.exposedHeaders || [
        'X-RateLimit-Limit',
        'X-RateLimit-Remaining',
        'X-RateLimit-Reset',
        'X-Request-ID'
      ],
      credentials: options.credentials !== undefined ? options.credentials : true,
      maxAge: options.maxAge || 86400,
      preflightContinue: options.preflightContinue || false,
      optionsSuccessStatus: options.optionsSuccessStatus || 204
    };

    this.allowedOrigins = new Set();
    if (Array.isArray(this.options.origin)) {
      this.options.origin.forEach(origin => this.allowedOrigins.add(origin));
    }
  }

  middleware() {
    return (req: Request, res: Response, next: NextFunction) => {
      const origin = req.headers.origin || req.headers.referer;

      if (this.isOriginAllowed(origin)) {
        res.setHeader('Access-Control-Allow-Origin', origin || '*');
        
        if (this.options.credentials) {
          res.setHeader('Access-Control-Allow-Credentials', 'true');
        }
      } else if (this.options.origin === '*') {
        res.setHeader('Access-Control-Allow-Origin', '*');
      }

      if (req.method === 'OPTIONS') {
        res.setHeader('Access-Control-Allow-Methods', this.options.methods.join(', '));
        res.setHeader('Access-Control-Allow-Headers', this.options.allowedHeaders.join(', '));
        res.setHeader('Access-Control-Max-Age', this.options.maxAge.toString());

        if (this.options.exposedHeaders.length > 0) {
          res.setHeader('Access-Control-Expose-Headers', this.options.exposedHeaders.join(', '));
        }

        if (!this.options.preflightContinue) {
          res.status(this.options.optionsSuccessStatus).end();
          return;
        }
      } else {
        if (this.options.exposedHeaders.length > 0) {
          res.setHeader('Access-Control-Expose-Headers', this.options.exposedHeaders.join(', '));
        }
      }

      next();
    };
  }

  private isOriginAllowed(origin: string | undefined): boolean {
    if (!origin) return false;

    if (typeof this.options.origin === 'function') {
      return this.options.origin(origin);
    }

    if (Array.isArray(this.options.origin)) {
      return this.allowedOrigins.has(origin);
    }

    if (typeof this.options.origin === 'string') {
      if (this.options.origin === '*') return true;
      return this.options.origin === origin;
    }

    return false;
  }

  addAllowedOrigin(origin: string): void {
    this.allowedOrigins.add(origin);
  }

  removeAllowedOrigin(origin: string): void {
    this.allowedOrigins.delete(origin);
  }

  getAllowedOrigins(): string[] {
    return Array.from(this.allowedOrigins);
  }
}

export const createCorsHandler = (options?: CorsOptions): CorsHandler => {
  return new CorsHandler(options);
};

export const createDynamicCorsHandler = () => {
  const handler = new CorsHandler({
    origin: (origin: string) => {
      const allowedPatterns = [
        /^https?:\/\/localhost(:\d+)?$/,
        /^https?:\/\/.*\.example\.com$/,
        /^https?:\/\/.*\.frauddetection\.io$/
      ];

      return allowedPatterns.some(pattern => pattern.test(origin));
    },
    credentials: true
  });

  return handler;
};
