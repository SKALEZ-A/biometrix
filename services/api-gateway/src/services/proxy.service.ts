import axios, { AxiosRequestConfig } from 'axios';
import { Request, Response } from 'express';
import { loadBalancer } from './load-balancer';
import { logger } from '@shared/utils/logger';
import { metricsCollector } from '@shared/utils/metrics';

export class ProxyService {
  async forwardRequest(req: Request, res: Response, serviceName: string): Promise<void> {
    const startTime = Date.now();
    const serviceUrl = loadBalancer.getServiceUrl(serviceName);

    if (!serviceUrl) {
      res.status(503).json({
        error: 'Service Unavailable',
        message: `${serviceName} is currently unavailable`
      });
      return;
    }

    try {
      const config: AxiosRequestConfig = {
        method: req.method as any,
        url: `${serviceUrl}${req.path}`,
        headers: this.prepareHeaders(req),
        params: req.query,
        data: req.body,
        timeout: 30000,
        validateStatus: () => true // Accept all status codes
      };

      logger.debug('Forwarding request', {
        serviceName,
        method: req.method,
        path: req.path,
        serviceUrl
      });

      const response = await axios(config);
      const duration = Date.now() - startTime;

      // Record metrics
      metricsCollector.recordHttpRequest(serviceName, req.method, response.status, duration);
      loadBalancer.updateResponseTime(serviceName, duration);

      // Forward response
      res.status(response.status)
        .set(this.prepareResponseHeaders(response.headers))
        .send(response.data);

      logger.debug('Request forwarded successfully', {
        serviceName,
        statusCode: response.status,
        duration
      });

    } catch (error: any) {
      const duration = Date.now() - startTime;
      
      loadBalancer.recordFailure(serviceName);
      metricsCollector.recordHttpRequest(serviceName, req.method, 500, duration);

      logger.error('Request forwarding failed', {
        serviceName,
        error: error.message,
        duration
      });

      if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
        res.status(503).json({
          error: 'Service Unavailable',
          message: `${serviceName} is not responding`
        });
      } else {
        res.status(500).json({
          error: 'Internal Server Error',
          message: 'An error occurred while processing your request'
        });
      }
    }
  }

  private prepareHeaders(req: Request): Record<string, string> {
    const headers: Record<string, string> = {};
    
    // Forward relevant headers
    const headersToForward = [
      'content-type',
      'authorization',
      'x-request-id',
      'x-correlation-id',
      'x-user-id',
      'x-tenant-id',
      'accept',
      'accept-language'
    ];

    headersToForward.forEach(header => {
      const value = req.headers[header];
      if (value) {
        headers[header] = Array.isArray(value) ? value[0] : value;
      }
    });

    // Add gateway-specific headers
    headers['x-forwarded-for'] = req.ip || req.socket.remoteAddress || '';
    headers['x-forwarded-proto'] = req.protocol;
    headers['x-forwarded-host'] = req.hostname;
    headers['x-gateway-timestamp'] = new Date().toISOString();

    return headers;
  }

  private prepareResponseHeaders(headers: Record<string, any>): Record<string, string> {
    const responseHeaders: Record<string, string> = {};
    
    // Forward relevant response headers
    const headersToForward = [
      'content-type',
      'content-length',
      'cache-control',
      'etag',
      'last-modified',
      'x-ratelimit-limit',
      'x-ratelimit-remaining',
      'x-ratelimit-reset'
    ];

    headersToForward.forEach(header => {
      if (headers[header]) {
        responseHeaders[header] = headers[header];
      }
    });

    return responseHeaders;
  }
}

export const proxyService = new ProxyService();
