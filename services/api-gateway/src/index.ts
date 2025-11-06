import express, { Application, Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import compression from 'compression';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { authMiddleware } from './middleware/auth.middleware';
import { rateLimiter } from './middleware/rate-limiter';
import { circuitBreaker } from './middleware/circuit-breaker';
import { loggingMiddleware } from './middleware/logging.middleware';
import { metricsMiddleware } from './middleware/metrics.middleware';
import { corsHandler } from './middleware/cors-handler';
import { requestValidator } from './middleware/request-validator';
import { gatewayConfig } from './config/gateway.config';
import { serviceDiscovery } from './services/service-discovery';
import { loadBalancer } from './services/load-balancer';
import { logger } from '@shared/utils/logger';

class APIGateway {
  private app: Application;
  private port: number;

  constructor() {
    this.app = express();
    this.port = gatewayConfig.port || 3000;
    this.initializeMiddlewares();
    this.initializeRoutes();
    this.initializeErrorHandling();
  }

  private initializeMiddlewares(): void {
    this.app.use(helmet());
    this.app.use(compression());
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
    this.app.use(corsHandler);
    this.app.use(loggingMiddleware);
    this.app.use(metricsMiddleware);
  }

  private initializeRoutes(): void {
    this.app.get('/health', (req: Request, res: Response) => {
      res.status(200).json({ status: 'healthy', timestamp: new Date().toISOString() });
    });

    this.app.use('/api/v1/biometric', 
      authMiddleware,
      rateLimiter({ maxRequests: 100, windowMs: 60000 }),
      circuitBreaker({ threshold: 5, timeout: 30000 }),
      requestValidator,
      this.createServiceProxy('biometric-service')
    );

    this.app.use('/api/v1/fraud-detection',
      authMiddleware,
      rateLimiter({ maxRequests: 200, windowMs: 60000 }),
      circuitBreaker({ threshold: 10, timeout: 60000 }),
      requestValidator,
      this.createServiceProxy('fraud-detection-service')
    );

    this.app.use('/api/v1/transactions',
      authMiddleware,
      rateLimiter({ maxRequests: 500, windowMs: 60000 }),
      circuitBreaker({ threshold: 15, timeout: 45000 }),
      requestValidator,
      this.createServiceProxy('transaction-service')
    );

    this.app.use('/api/v1/alerts',
      authMiddleware,
      rateLimiter({ maxRequests: 150, windowMs: 60000 }),
      this.createServiceProxy('alert-service')
    );

    this.app.use('/api/v1/compliance',
      authMiddleware,
      rateLimiter({ maxRequests: 100, windowMs: 60000 }),
      this.createServiceProxy('compliance-service')
    );

    this.app.use('/api/v1/analytics',
      authMiddleware,
      rateLimiter({ maxRequests: 50, windowMs: 60000 }),
      this.createServiceProxy('analytics-service')
    );

    this.app.use('/api/v1/users',
      authMiddleware,
      rateLimiter({ maxRequests: 200, windowMs: 60000 }),
      this.createServiceProxy('user-management-service')
    );

    this.app.use('/api/v1/merchants',
      authMiddleware,
      rateLimiter({ maxRequests: 100, windowMs: 60000 }),
      this.createServiceProxy('merchant-protection-service')
    );

    this.app.use('/api/v1/voice',
      authMiddleware,
      rateLimiter({ maxRequests: 80, windowMs: 60000 }),
      this.createServiceProxy('voice-service')
    );

    this.app.use('/api/v1/audit',
      authMiddleware,
      rateLimiter({ maxRequests: 100, windowMs: 60000 }),
      this.createServiceProxy('audit-service')
    );
  }

  private createServiceProxy(serviceName: string) {
    return createProxyMiddleware({
      target: serviceDiscovery.getServiceUrl(serviceName),
      changeOrigin: true,
      pathRewrite: (path) => path.replace(/^\/api\/v1\/[^/]+/, ''),
      onProxyReq: (proxyReq, req: any) => {
        if (req.user) {
          proxyReq.setHeader('X-User-Id', req.user.id);
          proxyReq.setHeader('X-User-Role', req.user.role);
        }
      },
      onError: (err, req, res: any) => {
        logger.error(`Proxy error for ${serviceName}:`, err);
        res.status(502).json({ error: 'Service unavailable', service: serviceName });
      }
    });
  }

  private initializeErrorHandling(): void {
    this.app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
      logger.error('Unhandled error:', err);
      res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? err.message : undefined
      });
    });

    this.app.use((req: Request, res: Response) => {
      res.status(404).json({ error: 'Route not found' });
    });
  }

  public async start(): Promise<void> {
    await serviceDiscovery.initialize();
    
    this.app.listen(this.port, () => {
      logger.info(`API Gateway running on port ${this.port}`);
      logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
    });
  }
}

const gateway = new APIGateway();
gateway.start().catch((error) => {
  logger.error('Failed to start API Gateway:', error);
  process.exit(1);
});

export default gateway;
