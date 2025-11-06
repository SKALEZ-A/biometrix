import express, { Application, Request, Response } from 'express';
import helmet from 'helmet';
import compression from 'compression';
import { auditRoutes } from './routes/audit.routes';
import { authMiddleware } from './middleware/auth.middleware';
import { errorHandlerMiddleware } from './middleware/error-handler.middleware';
import { rateLimitMiddleware } from './middleware/rate-limit.middleware';
import { appConfig } from './config/app.config';
import { logger } from '@shared/utils/logger';
import { connectDatabase } from '@shared/database/mongodb-client';

class AuditService {
  private app: Application;
  private port: number;

  constructor() {
    this.app = express();
    this.port = appConfig.port || 3009;
    this.initializeMiddlewares();
    this.initializeRoutes();
    this.initializeErrorHandling();
  }

  private initializeMiddlewares(): void {
    this.app.use(helmet());
    this.app.use(compression());
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
    this.app.use(rateLimitMiddleware);
  }

  private initializeRoutes(): void {
    this.app.get('/health', (req: Request, res: Response) => {
      res.status(200).json({
        status: 'healthy',
        service: 'audit-service',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
      });
    });

    this.app.use('/api/audit', authMiddleware, auditRoutes);
  }

  private initializeErrorHandling(): void {
    this.app.use(errorHandlerMiddleware);
    this.app.use((req: Request, res: Response) => {
      res.status(404).json({ error: 'Endpoint not found' });
    });
  }

  public async start(): Promise<void> {
    try {
      await connectDatabase();
      logger.info('Database connected successfully');

      this.app.listen(this.port, () => {
        logger.info(`Audit Service running on port ${this.port}`);
      });

      this.setupGracefulShutdown();
    } catch (error) {
      logger.error('Failed to start Audit Service:', error);
      process.exit(1);
    }
  }

  private setupGracefulShutdown(): void {
    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, shutting down gracefully');
      process.exit(0);
    });
  }
}

const service = new AuditService();
service.start();

export default service;
