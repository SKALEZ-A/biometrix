import { Request, Response, NextFunction } from 'express';
import { metrics } from '@shared/utils/metrics';

export const metricsMiddleware = (req: Request, res: Response, next: NextFunction): void => {
  const startTime = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const route = req.route?.path || req.path;
    const method = req.method;
    const statusCode = res.statusCode;

    metrics.recordHttpRequest({
      method,
      route,
      statusCode,
      duration,
      service: 'biometric-service',
    });

    metrics.incrementCounter('http_requests_total', {
      method,
      route,
      status: statusCode.toString(),
    });

    metrics.recordHistogram('http_request_duration_ms', duration, {
      method,
      route,
    });
  });

  next();
};
