import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { authMiddleware } from './middleware/auth';
import eventsRouter from './routes/events';
import profileRouter from './routes/profile';
import { errorHandler } from './middleware/errorHandler';
import { logger } from './utils/logger';
import { biometricProcessor } from './processors/biometricProcessor';
import { config } from './config/config';

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware setup
app.use(helmet()); // Security headers
app.use(cors({ origin: config.allowedOrigins }));
app.use(morgan('combined', { stream: { write: msg => logger.info(msg.trim()) } }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Health check
app.get('/health', (req: Request, res: Response) => {
  res.status(200).json({ status: 'healthy', timestamp: new Date().toISOString(), uptime: process.uptime() });
});

// API versioned routes
app.use('/api/v1/biometric/events', authMiddleware, eventsRouter);
app.use('/api/v1/biometric/profile', authMiddleware, profileRouter);

// Global error handler
app.use(errorHandler);

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
  });
});

const server = app.listen(PORT, () => {
  logger.info(`Biometric service running on port ${PORT}`);
  biometricProcessor.startBackgroundTasks();
});

export { app, server };
