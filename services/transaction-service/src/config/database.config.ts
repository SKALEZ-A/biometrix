import { MongoClient, Db } from 'mongodb';
import { createClient } from 'redis';
import neo4j, { Driver, Session } from 'neo4j-driver';

export interface TransactionDatabaseConfig {
  mongodb: {
    uri: string;
    database: string;
    collections: {
      transactions: string;
      riskScores: string;
      fraudCases: string;
      merchants: string;
    };
  };
  redis: {
    host: string;
    port: number;
    password?: string;
    db: number;
    keyPrefix: string;
  };
  neo4j: {
    uri: string;
    username: string;
    password: string;
    database: string;
    maxConnectionPoolSize: number;
  };
}

export const transactionDbConfig: TransactionDatabaseConfig = {
  mongodb: {
    uri: process.env.MONGODB_URI || 'mongodb://localhost:27017',
    database: process.env.MONGODB_DATABASE || 'transaction_service',
    collections: {
      transactions: 'transactions',
      riskScores: 'risk_scores',
      fraudCases: 'fraud_cases',
      merchants: 'merchants',
    },
  },
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD,
    db: parseInt(process.env.REDIS_DB || '1'),
    keyPrefix: 'txn:',
  },
  neo4j: {
    uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
    username: process.env.NEO4J_USERNAME || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'password',
    database: process.env.NEO4J_DATABASE || 'fraud-network',
    maxConnectionPoolSize: 50,
  },
};

class TransactionDatabaseManager {
  private mongoClient: MongoClient | null = null;
  private mongodb: Db | null = null;
  private redisClient: any = null;
  private neo4jDriver: Driver | null = null;

  async connectMongoDB(): Promise<Db> {
    if (this.mongodb) {
      return this.mongodb;
    }

    try {
      this.mongoClient = new MongoClient(transactionDbConfig.mongodb.uri, {
        maxPoolSize: 100,
        minPoolSize: 10,
      });
      await this.mongoClient.connect();
      this.mongodb = this.mongoClient.db(transactionDbConfig.mongodb.database);

      // Create indexes
      await this.createMongoIndexes();

      console.log('✅ Transaction MongoDB connected');
      return this.mongodb;
    } catch (error) {
      console.error('❌ Transaction MongoDB connection failed:', error);
      throw error;
    }
  }

  private async createMongoIndexes(): Promise<void> {
    if (!this.mongodb) return;

    const { transactions, riskScores, fraudCases, merchants } = transactionDbConfig.mongodb.collections;

    // Transaction indexes
    await this.mongodb.collection(transactions).createIndexes([
      { key: { userId: 1, timestamp: -1 } },
      { key: { merchantId: 1, timestamp: -1 } },
      { key: { transactionId: 1 }, unique: true },
      { key: { status: 1, timestamp: -1 } },
      { key: { amount: 1 } },
      { key: { 'geolocation.country': 1 } },
    ]);

    // Risk score indexes
    await this.mongodb.collection(riskScores).createIndexes([
      { key: { transactionId: 1 }, unique: true },
      { key: { userId: 1, timestamp: -1 } },
      { key: { riskScore: -1 } },
      { key: { decision: 1, timestamp: -1 } },
    ]);

    // Fraud case indexes
    await this.mongodb.collection(fraudCases).createIndexes([
      { key: { caseId: 1 }, unique: true },
      { key: { userId: 1, status: 1 } },
      { key: { createdAt: -1 } },
      { key: { fraudType: 1 } },
    ]);

    // Merchant indexes
    await this.mongodb.collection(merchants).createIndexes([
      { key: { merchantId: 1 }, unique: true },
      { key: { category: 1 } },
      { key: { riskLevel: 1 } },
    ]);

    console.log('✅ MongoDB indexes created');
  }

  async connectRedis(): Promise<any> {
    if (this.redisClient) {
      return this.redisClient;
    }

    try {
      this.redisClient = createClient({
        socket: {
          host: transactionDbConfig.redis.host,
          port: transactionDbConfig.redis.port,
        },
        password: transactionDbConfig.redis.password,
        database: transactionDbConfig.redis.db,
      });

      await this.redisClient.connect();
      console.log('✅ Transaction Redis connected');
      return this.redisClient;
    } catch (error) {
      console.error('❌ Transaction Redis connection failed:', error);
      throw error;
    }
  }

  connectNeo4j(): Driver {
    if (this.neo4jDriver) {
      return this.neo4jDriver;
    }

    try {
      this.neo4jDriver = neo4j.driver(
        transactionDbConfig.neo4j.uri,
        neo4j.auth.basic(
          transactionDbConfig.neo4j.username,
          transactionDbConfig.neo4j.password
        ),
        {
          maxConnectionPoolSize: transactionDbConfig.neo4j.maxConnectionPoolSize,
        }
      );

      console.log('✅ Neo4j connected');
      return this.neo4jDriver;
    } catch (error) {
      console.error('❌ Neo4j connection failed:', error);
      throw error;
    }
  }

  getNeo4jSession(): Session {
    if (!this.neo4jDriver) {
      throw new Error('Neo4j not connected');
    }
    return this.neo4jDriver.session({
      database: transactionDbConfig.neo4j.database,
    });
  }

  async disconnectAll(): Promise<void> {
    const promises: Promise<void>[] = [];

    if (this.mongoClient) {
      promises.push(this.mongoClient.close());
    }

    if (this.redisClient) {
      promises.push(this.redisClient.quit());
    }

    if (this.neo4jDriver) {
      promises.push(this.neo4jDriver.close());
    }

    await Promise.all(promises);
    console.log('✅ All transaction database connections closed');
  }

  getMongoDB(): Db {
    if (!this.mongodb) {
      throw new Error('MongoDB not connected');
    }
    return this.mongodb;
  }

  getRedis(): any {
    if (!this.redisClient) {
      throw new Error('Redis not connected');
    }
    return this.redisClient;
  }

  getNeo4jDriver(): Driver {
    if (!this.neo4jDriver) {
      throw new Error('Neo4j not connected');
    }
    return this.neo4jDriver;
  }
}

export const txnDbManager = new TransactionDatabaseManager();
