import { Pool, PoolClient, QueryResult } from 'pg';
import { logger } from '../utils/logger';

interface PostgresConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
  max?: number;
  idleTimeoutMillis?: number;
  connectionTimeoutMillis?: number;
  ssl?: boolean;
}

export class PostgresClient {
  private pool: Pool | null = null;
  private config: PostgresConfig;

  constructor(config: PostgresConfig) {
    this.config = {
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
      ssl: false,
      ...config
    };
  }

  async connect(): Promise<void> {
    try {
      this.pool = new Pool({
        host: this.config.host,
        port: this.config.port,
        database: this.config.database,
        user: this.config.user,
        password: this.config.password,
        max: this.config.max,
        idleTimeoutMillis: this.config.idleTimeoutMillis,
        connectionTimeoutMillis: this.config.connectionTimeoutMillis,
        ssl: this.config.ssl ? { rejectUnauthorized: false } : false
      });

      // Test connection
      const client = await this.pool.connect();
      await client.query('SELECT NOW()');
      client.release();

      logger.info('PostgreSQL connected successfully', {
        host: this.config.host,
        database: this.config.database
      });
    } catch (error) {
      logger.error('PostgreSQL connection failed', { error });
      throw error;
    }
  }

  async query<T = any>(text: string, params?: any[]): Promise<QueryResult<T>> {
    if (!this.pool) {
      throw new Error('PostgreSQL pool not initialized');
    }

    const start = Date.now();
    try {
      const result = await this.pool.query<T>(text, params);
      const duration = Date.now() - start;
      
      logger.debug('Query executed', { duration, rows: result.rowCount });
      return result;
    } catch (error) {
      logger.error('Query execution failed', { error, query: text });
      throw error;
    }
  }

  async getClient(): Promise<PoolClient> {
    if (!this.pool) {
      throw new Error('PostgreSQL pool not initialized');
    }
    return this.pool.connect();
  }

  async transaction<T>(callback: (client: PoolClient) => Promise<T>): Promise<T> {
    const client = await this.getClient();
    
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      logger.error('Transaction failed', { error });
      throw error;
    } finally {
      client.release();
    }
  }

  async findOne<T = any>(table: string, conditions: Record<string, any>): Promise<T | null> {
    const keys = Object.keys(conditions);
    const values = Object.values(conditions);
    const whereClause = keys.map((key, index) => `${key} = $${index + 1}`).join(' AND ');
    
    const query = `SELECT * FROM ${table} WHERE ${whereClause} LIMIT 1`;
    const result = await this.query<T>(query, values);
    
    return result.rows[0] || null;
  }

  async findMany<T = any>(
    table: string,
    conditions?: Record<string, any>,
    options?: { limit?: number; offset?: number; orderBy?: string }
  ): Promise<T[]> {
    let query = `SELECT * FROM ${table}`;
    const values: any[] = [];

    if (conditions && Object.keys(conditions).length > 0) {
      const keys = Object.keys(conditions);
      const whereClause = keys.map((key, index) => {
        values.push(conditions[key]);
        return `${key} = $${index + 1}`;
      }).join(' AND ');
      query += ` WHERE ${whereClause}`;
    }

    if (options?.orderBy) {
      query += ` ORDER BY ${options.orderBy}`;
    }

    if (options?.limit) {
      query += ` LIMIT ${options.limit}`;
    }

    if (options?.offset) {
      query += ` OFFSET ${options.offset}`;
    }

    const result = await this.query<T>(query, values);
    return result.rows;
  }

  async insert<T = any>(table: string, data: Record<string, any>): Promise<T> {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const placeholders = keys.map((_, index) => `$${index + 1}`).join(', ');
    
    const query = `
      INSERT INTO ${table} (${keys.join(', ')})
      VALUES (${placeholders})
      RETURNING *
    `;
    
    const result = await this.query<T>(query, values);
    return result.rows[0];
  }

  async update<T = any>(
    table: string,
    conditions: Record<string, any>,
    data: Record<string, any>
  ): Promise<T[]> {
    const dataKeys = Object.keys(data);
    const dataValues = Object.values(data);
    const conditionKeys = Object.keys(conditions);
    const conditionValues = Object.values(conditions);

    const setClause = dataKeys.map((key, index) => `${key} = $${index + 1}`).join(', ');
    const whereClause = conditionKeys.map((key, index) => 
      `${key} = $${dataKeys.length + index + 1}`
    ).join(' AND ');

    const query = `
      UPDATE ${table}
      SET ${setClause}
      WHERE ${whereClause}
      RETURNING *
    `;

    const result = await this.query<T>(query, [...dataValues, ...conditionValues]);
    return result.rows;
  }

  async delete(table: string, conditions: Record<string, any>): Promise<number> {
    const keys = Object.keys(conditions);
    const values = Object.values(conditions);
    const whereClause = keys.map((key, index) => `${key} = $${index + 1}`).join(' AND ');
    
    const query = `DELETE FROM ${table} WHERE ${whereClause}`;
    const result = await this.query(query, values);
    
    return result.rowCount || 0;
  }

  async count(table: string, conditions?: Record<string, any>): Promise<number> {
    let query = `SELECT COUNT(*) as count FROM ${table}`;
    const values: any[] = [];

    if (conditions && Object.keys(conditions).length > 0) {
      const keys = Object.keys(conditions);
      const whereClause = keys.map((key, index) => {
        values.push(conditions[key]);
        return `${key} = $${index + 1}`;
      }).join(' AND ');
      query += ` WHERE ${whereClause}`;
    }

    const result = await this.query<{ count: string }>(query, values);
    return parseInt(result.rows[0].count, 10);
  }

  async disconnect(): Promise<void> {
    if (this.pool) {
      await this.pool.end();
      this.pool = null;
      logger.info('PostgreSQL disconnected');
    }
  }

  getPool(): Pool | null {
    return this.pool;
  }
}

export default PostgresClient;
