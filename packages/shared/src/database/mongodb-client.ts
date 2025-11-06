import { MongoClient, Db, Collection } from 'mongodb';
import { logger } from '../utils/logger';

export class MongoDBClient {
  private client: MongoClient;
  private db: Db | null = null;
  private static instance: MongoDBClient;

  private constructor() {
    const uri = process.env.MONGODB_URI || 'mongodb://localhost:27017';
    
    this.client = new MongoClient(uri, {
      maxPoolSize: parseInt(process.env.MONGODB_POOL_SIZE || '10'),
      minPoolSize: 2,
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000
    });
  }

  static getInstance(): MongoDBClient {
    if (!MongoDBClient.instance) {
      MongoDBClient.instance = new MongoDBClient();
    }
    return MongoDBClient.instance;
  }

  async connect(): Promise<void> {
    try {
      await this.client.connect();
      this.db = this.client.db(process.env.MONGODB_DB || 'fraud_prevention');
      logger.info('Connected to MongoDB');
    } catch (error) {
      logger.error('MongoDB connection failed', { error });
      throw error;
    }
  }

  getDatabase(): Db {
    if (!this.db) {
      throw new Error('Database not connected');
    }
    return this.db;
  }

  getCollection<T = any>(name: string): Collection<T> {
    return this.getDatabase().collection<T>(name);
  }

  async close(): Promise<void> {
    await this.client.close();
    logger.info('MongoDB connection closed');
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.client.db().admin().ping();
      return true;
    } catch (error) {
      logger.error('MongoDB health check failed', { error });
      return false;
    }
  }
}
