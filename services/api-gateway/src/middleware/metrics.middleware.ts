import { Request, Response, NextFunction } from 'express';
import { MonitoringUtils } from '../../../packages/shared/src/utils/monitoring.utils';

export class MetricsMiddleware {
  static collectMetrics(req: Request, res: Response, next: NextFunction) {
    const startTime = Date.now();
    const route = `${req.method} ${req.route?.path || req.path}`;

    res.on('finish', () => {
      const duration = Date.now() - startTime;

      MonitoringUtils.recordLatency(route, duration, {
        method: req.method,
        path: req.path,
        statusCode: res.statusCode.toString()
      });

      MonitoringUtils.recordThroughput(route, 1, {
        method: req.method,
        statusCode: res.statusCode.toString()
      });

      if (res.statusCode >= 400) {
        MonitoringUtils.recordMetric({
          name: `http.errors.${res.statusCode}`,
          value: 1,
          timestamp: new Date(),
          tags: {
            method: req.method,
            path: req.path
          }
        });
      }
    });

    next();
  }

  static async getMetrics(req: Request, res: Response) {
    const health = MonitoringUtils.getHealthStatus();
    res.json({ success: true, data: health });
  }
}
