import { MongoClient, Db } from 'mongodb';
import { createClient } from 'redis';
import { InfluxDB, Point } from '@influxdata/influxdb-client';

export interface DatabaseConfig {
  mongodb: {
    uri: string;
    database: string;
    options: {
      maxPoolSize: number;
      minPoolSize: number;
      serverSelectionTimeoutMS: number;
      socketTimeoutMS: number;
      retryWrites: boolean;
      w: string;
    };
  };
  redis: {
    host: string;
    port: number;
    password?: string;
    db: number;
    maxRetriesPerRequest: number;
    enableReadyCheck: boolean;
    connectTimeout: number;
  };
  influxdb: {
    url: string;
    token: string;
    org: string;
    bucket: string;
    timeout: number;
  };
}

export const databaseConfig: DatabaseConfig = {
  mongodb: {
    uri: process.env.MONGODB_URI || 'mongodb://localhost:27017',
    database: process.env.MONGODB_DATABASE || 'biometric_fraud_prevention',
    options: {
      maxPoolSize: 100,
      minPoolSize: 10,
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
      retryWrites: true,
      w: 'majority',
    },
  },
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD,
    db: parseInt(process.env.REDIS_DB || '0'),
    maxRetriesPerRequest: 3,
    enableReadyCheck: true,
    connectTimeout: 10000,
  },
  influxdb: {
    url: process.env.INFLUXDB_URL || 'http://localhost:8086',
    token: process.env.INFLUXDB_TOKEN || '',
    org: process.env.INFLUXDB_ORG || 'fraud-prevention',
    bucket: process.env.INFLUXDB_BUCKET || 'biometric-events',
    timeout: 30000,
  },
};

class DatabaseManager {
  private mongoClient: MongoClient | null = null;
  private mongodb: Db | null = null;
  private redisClient: any = null;
  private influxClient: InfluxDB | null = null;

  async connectMongoDB(): Promise<Db> {
    if (this.mongodb) {
      return this.mongodb;
    }

    try {
      this.mongoClient = new MongoClient(
        databaseConfig.mongodb.uri,
        databaseConfig.mongodb.options
      );
      await this.mongoClient.connect();
      this.mongodb = this.mongoClient.db(databaseConfig.mongodb.database);
      console.log('✅ MongoDB connected successfully');
      return this.mongodb;
    } catch (error) {
      console.error('❌ MongoDB connection failed:', error);
      throw error;
    }
  }

  async connectRedis(): Promise<any> {
    if (this.redisClient) {
      return this.redisClient;
    }

    try {
      this.redisClient = createClient({
        socket: {
          host: databaseConfig.redis.host,
          port: databaseConfig.redis.port,
          connectTimeout: databaseConfig.redis.connectTimeout,
        },
        password: databaseConfig.redis.password,
        database: databaseConfig.redis.db,
      });

      this.redisClient.on('error', (err: Error) => {
        console.error('Redis Client Error:', err);
      });

      this.redisClient.on('connect', () => {
        console.log('✅ Redis connected successfully');
      });

      await this.redisClient.connect();
      return this.redisClient;
    } catch (error) {
      console.error('❌ Redis connection failed:', error);
      throw error;
    }
  }

  connectInfluxDB(): InfluxDB {
    if (this.influxClient) {
      return this.influxClient;
    }

    try {
      this.influxClient = new InfluxDB({
        url: databaseConfig.influxdb.url,
        token: databaseConfig.influxdb.token,
        timeout: databaseConfig.influxdb.timeout,
      });
      console.log('✅ InfluxDB connected successfully');
      return this.influxClient;
    } catch (error) {
      console.error('❌ InfluxDB connection failed:', error);
      throw error;
    }
  }

  async disconnectAll(): Promise<void> {
    const promises: Promise<void>[] = [];

    if (this.mongoClient) {
      promises.push(this.mongoClient.close());
    }

    if (this.redisClient) {
      promises.push(this.redisClient.quit());
    }

    await Promise.all(promises);
    console.log('✅ All database connections closed');
  }

  getMongoDB(): Db {
    if (!this.mongodb) {
      throw new Error('MongoDB not connected. Call connectMongoDB() first.');
    }
    return this.mongodb;
  }

  getRedis(): any {
    if (!this.redisClient) {
      throw new Error('Redis not connected. Call connectRedis() first.');
    }
    return this.redisClient;
  }

  getInfluxDB(): InfluxDB {
    if (!this.influxClient) {
      throw new Error('InfluxDB not connected. Call connectInfluxDB() first.');
    }
    return this.influxClient;
  }
}

export const dbManager = new DatabaseManager();
