import express, { Application, Request, Response } from 'express';
import helmet from 'helmet';
import compression from 'compression';
import { userRoutes } from './routes/user.routes';
import { authRoutes } from './routes/auth.routes';
import { roleRoutes } from './routes/role.routes';
import { permissionRoutes } from './routes/permission.routes';
import { authMiddleware } from './middleware/auth.middleware';
import { errorHandlerMiddleware } from './middleware/error-handler.middleware';
import { rateLimitMiddleware } from './middleware/rate-limit.middleware';
import { appConfig } from './config/app.config';
import { logger } from '@shared/utils/logger';
import { connectDatabase } from '@shared/database/postgres-client';
import { initializeRedis } from '@shared/cache/redis';

class UserManagementService {
  private app: Application;
  private port: number;

  constructor() {
    this.app = express();
    this.port = appConfig.port || 3006;
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
        service: 'user-management-service',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
      });
    });

    this.app.use('/api/auth', authRoutes);
    this.app.use('/api/users', authMiddleware, userRoutes);
    this.app.use('/api/roles', authMiddleware, roleRoutes);
    this.app.use('/api/permissions', authMiddleware, permissionRoutes);
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
      await initializeRedis();
      logger.info('All dependencies initialized successfully');

      this.app.listen(this.port, () => {
        logger.info(`User Management Service running on port ${this.port}`);
      });

      this.setupGracefulShutdown();
    } catch (error) {
      logger.error('Failed to start User Management Service:', error);
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

const service = new UserManagementService();
service.start();

export default service;
