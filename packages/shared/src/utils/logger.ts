import winston from 'winston';
import DailyRotateFile from 'winston-daily-rotate-file';
import path from 'path';

interface LogMetadata {
  service?: string;
  userId?: string;
  transactionId?: string;
  requestId?: string;
  [key: string]: any;
}

class Logger {
  private logger: winston.Logger;
  private serviceName: string;

  constructor(serviceName: string = 'default-service') {
    this.serviceName = serviceName;
    this.logger = this.createLogger();
  }

  private createLogger(): winston.Logger {
    const logFormat = winston.format.combine(
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      winston.format.errors({ stack: true }),
      winston.format.splat(),
      winston.format.json(),
      winston.format.printf(({ timestamp, level, message, service, ...metadata }) => {
        let msg = `${timestamp} [${level.toUpperCase()}] [${service || this.serviceName}]: ${message}`;
        
        if (Object.keys(metadata).length > 0) {
          msg += ` ${JSON.stringify(metadata)}`;
        }
        
        return msg;
      })
    );

    const transports: winston.transport[] = [
      new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple()
        )
      })
    ];

    if (process.env.NODE_ENV === 'production') {
      transports.push(
        new DailyRotateFile({
          filename: path.join('logs', '%DATE%-error.log'),
          datePattern: 'YYYY-MM-DD',
          level: 'error',
          maxSize: '20m',
          maxFiles: '14d',
          format: logFormat
        }),
        new DailyRotateFile({
          filename: path.join('logs', '%DATE%-combined.log'),
          datePattern: 'YYYY-MM-DD',
          maxSize: '20m',
          maxFiles: '14d',
          format: logFormat
        })
      );
    }

    return winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: logFormat,
      defaultMeta: { service: this.serviceName },
      transports,
      exceptionHandlers: [
        new winston.transports.File({ filename: path.join('logs', 'exceptions.log') })
      ],
      rejectionHandlers: [
        new winston.transports.File({ filename: path.join('logs', 'rejections.log') })
      ]
    });
  }

  public info(message: string, metadata?: LogMetadata): void {
    this.logger.info(message, metadata);
  }

  public error(message: string, error?: Error | any, metadata?: LogMetadata): void {
    if (error instanceof Error) {
      this.logger.error(message, {
        ...metadata,
        error: {
          message: error.message,
          stack: error.stack,
          name: error.name
        }
      });
    } else {
      this.logger.error(message, { ...metadata, error });
    }
  }

  public warn(message: string, metadata?: LogMetadata): void {
    this.logger.warn(message, metadata);
  }

  public debug(message: string, metadata?: LogMetadata): void {
    this.logger.debug(message, metadata);
  }

  public verbose(message: string, metadata?: LogMetadata): void {
    this.logger.verbose(message, metadata);
  }

  public http(message: string, metadata?: LogMetadata): void {
    this.logger.http(message, metadata);
  }

  public logTransaction(transactionId: string, action: string, metadata?: LogMetadata): void {
    this.info(`Transaction ${action}`, {
      transactionId,
      action,
      ...metadata
    });
  }

  public logFraudDetection(transactionId: string, score: number, isF raud: boolean, metadata?: LogMetadata): void {
    this.info('Fraud detection completed', {
      transactionId,
      fraudScore: score,
      isFraud,
      ...metadata
    });
  }

  public logBiometricVerification(userId: string, biometricType: string, success: boolean, metadata?: LogMetadata): void {
    this.info('Biometric verification', {
      userId,
      biometricType,
      success,
      ...metadata
    });
  }

  public logAPIRequest(method: string, path: string, statusCode: number, duration: number, metadata?: LogMetadata): void {
    this.http('API Request', {
      method,
      path,
      statusCode,
      duration,
      ...metadata
    });
  }

  public logDatabaseQuery(query: string, duration: number, metadata?: LogMetadata): void {
    this.debug('Database query', {
      query,
      duration,
      ...metadata
    });
  }

  public logCacheOperation(operation: string, key: string, hit: boolean, metadata?: LogMetadata): void {
    this.debug('Cache operation', {
      operation,
      key,
      hit,
      ...metadata
    });
  }

  public logMLPrediction(modelName: string, prediction: any, confidence: number, metadata?: LogMetadata): void {
    this.info('ML prediction', {
      modelName,
      prediction,
      confidence,
      ...metadata
    });
  }

  public logSecurityEvent(eventType: string, severity: string, metadata?: LogMetadata): void {
    this.warn('Security event', {
      eventType,
      severity,
      ...metadata
    });
  }

  public logPerformanceMetric(metricName: string, value: number, unit: string, metadata?: LogMetadata): void {
    this.info('Performance metric', {
      metricName,
      value,
      unit,
      ...metadata
    });
  }
}

export const logger = new Logger(process.env.SERVICE_NAME || 'fraud-detection-platform');
export default Logger;
