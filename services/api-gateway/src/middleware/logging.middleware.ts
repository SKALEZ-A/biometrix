import { Request, Response, NextFunction } from 'express';
import { logger } from '../../../packages/shared/src/utils/logger';
import { v4 as uuidv4 } from 'uuid';

export interface RequestLog {
  requestId: string;
  method: string;
  path: string;
  query: any;
  body: any;
  headers: any;
  ip: string;
  userAgent: string;
  timestamp: Date;
  duration?: number;
  statusCode?: number;
  error?: any;
}

export class LoggingMiddleware {
  static requestLogger(req: Request, res: Response, next: NextFunction) {
    const requestId = uuidv4();
    const startTime = Date.now();

    req.headers['x-request-id'] = requestId;

    const requestLog: RequestLog = {
      requestId,
      method: req.method,
      path: req.path,
      query: req.query,
      body: LoggingMiddleware.sanitizeBody(req.body),
      headers: LoggingMiddleware.sanitizeHeaders(req.headers),
      ip: req.ip || req.socket.remoteAddress || 'unknown',
      userAgent: req.get('user-agent') || 'unknown',
      timestamp: new Date()
    };

    logger.info('Incoming request', requestLog);

    const originalSend = res.send;
    res.send = function (data: any) {
      const duration = Date.now() - startTime;

      logger.info('Request completed', {
        requestId,
        statusCode: res.statusCode,
        duration,
        path: req.path
      });

      return originalSend.call(this, data);
    };

    res.on('finish', () => {
      const duration = Date.now() - startTime;

      if (res.statusCode >= 400) {
        logger.warn('Request failed', {
          requestId,
          statusCode: res.statusCode,
          duration,
          path: req.path
        });
      }
    });

    next();
  }

  private static sanitizeBody(body: any): any {
    if (!body) return body;

    const sanitized = { ...body };
    const sensitiveFields = ['password', 'token', 'apiKey', 'secret', 'creditCard'];

    for (const field of sensitiveFields) {
      if (sanitized[field]) {
        sanitized[field] = '***REDACTED***';
      }
    }

    return sanitized;
  }

  private static sanitizeHeaders(headers: any): any {
    const sanitized = { ...headers };
    const sensitiveHeaders = ['authorization', 'x-api-key', 'cookie'];

    for (const header of sensitiveHeaders) {
      if (sanitized[header]) {
        sanitized[header] = '***REDACTED***';
      }
    }

    return sanitized;
  }

  static errorLogger(err: Error, req: Request, res: Response, next: NextFunction) {
    const requestId = req.headers['x-request-id'] as string;

    logger.error('Request error', {
      requestId,
      error: {
        name: err.name,
        message: err.message,
        stack: err.stack
      },
      path: req.path,
      method: req.method
    });

    next(err);
  }
}
