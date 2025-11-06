export const appConfig = {
  port: process.env.PORT || 3003,
  env: process.env.NODE_ENV || 'development',
  
  ml: {
    xgboost: {
      endpoint: process.env.XGBOOST_ENDPOINT || 'http://localhost:5001',
      timeout: 3000,
      retries: 2
    },
    isolationForest: {
      endpoint: process.env.ISOLATION_FOREST_ENDPOINT || 'http://localhost:5002',
      timeout: 2000,
      retries: 2
    },
    lstm: {
      endpoint: process.env.LSTM_ENDPOINT || 'http://localhost:5003',
      timeout: 4000,
      retries: 1
    },
    deepfake: {
      endpoint: process.env.DEEPFAKE_ENDPOINT || 'http://localhost:5004',
      timeout: 5000,
      retries: 1
    }
  },

  rateLimit: {
    windowMs: 15 * 60 * 1000,
    max: 100
  },

  cors: {
    origin: process.env.CORS_ORIGIN || '*',
    credentials: true
  },

  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: 'json'
  }
};
