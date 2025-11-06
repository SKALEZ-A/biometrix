export const appConfig = {
  port: process.env.PORT || 3001,
  env: process.env.NODE_ENV || 'development',
  jwt: {
    secret: process.env.JWT_SECRET || 'your-secret-key',
    expiresIn: process.env.JWT_EXPIRES_IN || '24h',
    refreshExpiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d'
  },
  database: {
    mongodb: {
      uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/fraud_detection',
      options: {
        useNewUrlParser: true,
        useUnifiedTopology: true
      }
    }
  },
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD
  },
  security: {
    bcryptRounds: 10,
    maxLoginAttempts: 5,
    lockoutDuration: 15 * 60 * 1000,
    passwordMinLength: 8
  },
  cors: {
    origin: process.env.CORS_ORIGIN || '*',
    credentials: true
  }
};
