import { Pool } from 'pg';
import { Transaction, TransactionStatus } from '../models/transaction.model';

export class TransactionRepository {
  constructor(private pool: Pool) {}

  async create(transaction: Omit<Transaction, 'id' | 'createdAt' | 'updatedAt'>): Promise<Transaction> {
    const query = `
      INSERT INTO transactions (
        user_id, merchant_id, amount, currency, payment_method,
        status, device_fingerprint, ip_address, location, metadata
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      RETURNING *
    `;

    const values = [
      transaction.userId,
      transaction.merchantId,
      transaction.amount,
      transaction.currency,
      transaction.paymentMethod,
      transaction.status,
      transaction.deviceFingerprint,
      transaction.ipAddress,
      JSON.stringify(transaction.location),
      JSON.stringify(transaction.metadata)
    ];

    const result = await this.pool.query(query, values);
    return this.mapRowToTransaction(result.rows[0]);
  }

  async findById(transactionId: string): Promise<Transaction | null> {
    const query = 'SELECT * FROM transactions WHERE id = $1';
    const result = await this.pool.query(query, [transactionId]);
    if (result.rows.length === 0) return null;
    return this.mapRowToTransaction(result.rows[0]);
  }

  async findByUserId(userId: string, limit: number = 50): Promise<Transaction[]> {
    const query = 'SELECT * FROM transactions WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2';
    const result = await this.pool.query(query, [userId, limit]);
    return result.rows.map(this.mapRowToTransaction);
  }

  async updateStatus(transactionId: string, status: TransactionStatus): Promise<Transaction | null> {
    const query = `
      UPDATE transactions
      SET status = $1, updated_at = $2
      WHERE id = $3
      RETURNING *
    `;
    const result = await this.pool.query(query, [status, new Date(), transactionId]);
    if (result.rows.length === 0) return null;
    return this.mapRowToTransaction(result.rows[0]);
  }

  private mapRowToTransaction(row: any): Transaction {
    return {
      id: row.id,
      userId: row.user_id,
      merchantId: row.merchant_id,
      amount: parseFloat(row.amount),
      currency: row.currency,
      paymentMethod: row.payment_method,
      status: row.status,
      deviceFingerprint: row.device_fingerprint,
      ipAddress: row.ip_address,
      location: typeof row.location === 'string' ? JSON.parse(row.location) : row.location,
      metadata: typeof row.metadata === 'string' ? JSON.parse(row.metadata) : row.metadata,
      createdAt: row.created_at,
      updatedAt: row.updated_at
    };
  }
}
