import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { authMiddleware } from './middleware/auth';
import transactionRouter from './routes/transactions';
import { errorHandler } from './middleware/errorHandler';
import { logger } from './utils/logger';
import { decisionEngine } from './engines/decisionEngine';
import { fraudRules } from './rules/fraudRules';
import { config } from './config/config';
import { TransactionRiskRequest, TransactionDecision } from './types/transaction';

const app = express();
const PORT = process.env.PORT || 3002;

// Middleware
app.use(helmet());
app.use(cors({ origin: config.allowedOrigins }));
app.use(morgan('combined', { stream: { write: msg => logger.info(msg.trim()) } }));
app.use(express.json({ limit: '10mb' }));

// Health check with decision engine status
app.get('/health', async (req: Request, res: Response) => {
  const engineStatus = await decisionEngine.healthCheck();
  res.status(200).json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(), 
    uptime: process.uptime(),
    decisionEngine: engineStatus 
  });
});

// Routes
app.use('/api/v1/transactions', authMiddleware, transactionRouter);

// Error handling
app.use(errorHandler);

// Background task: Periodic rule updates and model refresh
let ruleRefreshInterval: NodeJS.Timeout;
const startBackgroundTasks = () => {
  ruleRefreshInterval = setInterval(async () => {
    await fraudRules.updateRulesFromSource();
    logger.info('Fraud rules refreshed');
  }, 30 * 60 * 1000); // Every 30 min

  // Simulate ML model warm-up
  decisionEngine.warmupModels();
};

process.on('SIGTERM', () => {
  logger.info('Shutting down transaction service');
  clearInterval(ruleRefreshInterval);
  server.close(() => logger.info('Server closed'));
});

const server = app.listen(PORT, () => {
  logger.info(`Transaction service running on port ${PORT}`);
  startBackgroundTasks();
});

export { app, server };

// Export for testing
export { decisionEngine, fraudRules };
