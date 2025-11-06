/**
 * Neo4j Database Client
 * Handles all interactions with Neo4j graph database
 */

import neo4j, { Driver, Session, Result, QueryResult, Integer } from 'neo4j-driver';
import winston from 'winston';

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [new winston.transports.Console()]
});

export interface Neo4jConfig {
  uri: string;
  user: string;
  password: string;
  database?: string;
  maxConnectionPoolSize?: number;
  connectionTimeout?: number;
}

export class Neo4jGraphDB {
  private driver: Driver | null = null;
  private config: Neo4jConfig;

  constructor(config: Neo4jConfig) {
    this.config = {
      ...config,
      database: config.database || 'neo4j',
      maxConnectionPoolSize: config.maxConnectionPoolSize || 50,
      connectionTimeout: config.connectionTimeout || 30000
    };
  }

  async connect(): Promise<void> {
    try {
      this.driver = neo4j.driver(
        this.config.uri,
        neo4j.auth.basic(this.config.user, this.config.password),
        {
          maxConnectionPoolSize: this.config.maxConnectionPoolSize,
          connectionTimeout: this.config.connectionTimeout
        }
      );

      // Verify connectivity
      await this.driver.verifyConnectivity();
      logger.info('Successfully connected to Neo4j');
    } catch (error) {
      logger.error('Failed to connect to Neo4j', { error });
      throw error;
    }
  }

  async run(query: string, parameters: Record<string, any> = {}): Promise<QueryResult> {
    if (!this.driver) {
      throw new Error('Database not connected');
    }

    const session = this.driver.session({ database: this.config.database });

    try {
      const result = await session.run(query, parameters);
      return result;
    } catch (error) {
      logger.error('Query execution failed', { error, query });
      throw error;
    } finally {
      await session.close();
    }
  }

  async close(): Promise<void> {
    if (this.driver) {
      await this.driver.close();
      logger.info('Neo4j connection closed');
    }
  }
}
