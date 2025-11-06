export interface AppConfig {
  server: {
    port: number;
    host: string;
    env: string;
    corsOrigins: string[];
    rateLimitWindowMs: number;
    rateLimitMaxRequests: number;
  };
  biometric: {
    keystroke: {
      minEventsForProfile: number;
      dwellTimeThreshold: number;
      flightTimeThreshold: number;
      matchThreshold: number;
    };
    mouse: {
      minEventsForProfile: number;
      velocityThreshold: number;
      accelerationThreshold: number;
      matchThreshold: number;
    };
    touch: {
      minEventsForProfile: number;
      pressureThreshold: number;
      areaThreshold: number;
      matchThreshold: number;
    };
  };
  security: {
    jwtSecret: string;
    jwtExpiresIn: string;
    encryptionKey: string;
    hashRounds: number;
  };
  monitoring: {
    enableMetrics: boolean;
    metricsPort: number;
    logLevel: string;
  };
}

export const appConfig: AppConfig = {
  server: {
    port: parseInt(process.env.PORT || '3001'),
    host: process.env.HOST || '0.0.0.0',
    env: process.env.NODE_ENV || 'development',
    corsOrigins: (process.env.CORS_ORIGINS || 'http://localhost:3000').split(','),
    rateLimitWindowMs: 15 * 60 * 1000, // 15 minutes
    rateLimitMaxRequests: 100,
  },
  biometric: {
    keystroke: {
      minEventsForProfile: 500,
      dwellTimeThreshold: 200,
      flightTimeThreshold: 150,
      matchThreshold: 0.85,
    },
    mouse: {
      minEventsForProfile: 1000,
      velocityThreshold: 500,
      accelerationThreshold: 1000,
      matchThreshold: 0.80,
    },
    touch: {
      minEventsForProfile: 300,
      pressureThreshold: 0.5,
      areaThreshold: 100,
      matchThreshold: 0.82,
    },
  },
  security: {
    jwtSecret: process.env.JWT_SECRET || 'your-secret-key-change-in-production',
    jwtExpiresIn: process.env.JWT_EXPIRES_IN || '24h',
    encryptionKey: process.env.ENCRYPTION_KEY || 'your-encryption-key-32-bytes-long',
    hashRounds: 12,
  },
  monitoring: {
    enableMetrics: process.env.ENABLE_METRICS === 'true',
    metricsPort: parseInt(process.env.METRICS_PORT || '9090'),
    logLevel: process.env.LOG_LEVEL || 'info',
  },
};
