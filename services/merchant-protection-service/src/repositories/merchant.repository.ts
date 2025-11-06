import { Pool } from 'pg';
import { Merchant, Chargeback, Dispute } from '../models/merchant.model';

export class MerchantRepository {
  constructor(private pool: Pool) {}

  async findById(merchantId: string): Promise<Merchant | null> {
    const query = 'SELECT * FROM merchants WHERE id = $1';
    const result = await this.pool.query(query, [merchantId]);

    if (result.rows.length === 0) return null;
    return this.mapRowToMerchant(result.rows[0]);
  }

  async updateRiskScore(merchantId: string, riskScore: number, riskLevel: string): Promise<Merchant | null> {
    const query = `
      UPDATE merchants
      SET risk_score = $1, risk_level = $2, updated_at = $3
      WHERE id = $4
      RETURNING *
    `;

    const result = await this.pool.query(query, [riskScore, riskLevel, new Date(), merchantId]);

    if (result.rows.length === 0) return null;
    return this.mapRowToMerchant(result.rows[0]);
  }

  async createChargeback(chargeback: Omit<Chargeback, 'id' | 'filedAt'>): Promise<Chargeback> {
    const query = `
      INSERT INTO chargebacks (
        transaction_id, merchant_id, amount, reason, status
      ) VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;

    const values = [
      chargeback.transactionId,
      chargeback.merchantId,
      chargeback.amount,
      chargeback.reason,
      chargeback.status
    ];

    const result = await this.pool.query(query, values);
    return this.mapRowToChargeback(result.rows[0]);
  }

  async findChargebacksByMerchantId(merchantId: string): Promise<Chargeback[]> {
    const query = 'SELECT * FROM chargebacks WHERE merchant_id = $1 ORDER BY filed_at DESC';
    const result = await this.pool.query(query, [merchantId]);
    return result.rows.map(this.mapRowToChargeback);
  }

  async createDispute(dispute: Omit<Dispute, 'id' | 'createdAt' | 'updatedAt'>): Promise<Dispute> {
    const query = `
      INSERT INTO disputes (
        chargeback_id, merchant_id, status, evidence, notes
      ) VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;

    const values = [
      dispute.chargebackId,
      dispute.merchantId,
      dispute.status,
      JSON.stringify(dispute.evidence),
      JSON.stringify(dispute.notes)
    ];

    const result = await this.pool.query(query, values);
    return this.mapRowToDispute(result.rows[0]);
  }

  async updateDisputeStatus(disputeId: string, status: string): Promise<Dispute | null> {
    const query = `
      UPDATE disputes
      SET status = $1, updated_at = $2
      WHERE id = $3
      RETURNING *
    `;

    const result = await this.pool.query(query, [status, new Date(), disputeId]);

    if (result.rows.length === 0) return null;
    return this.mapRowToDispute(result.rows[0]);
  }

  private mapRowToMerchant(row: any): Merchant {
    return {
      id: row.id,
      name: row.name,
      businessName: row.business_name,
      category: row.category,
      status: row.status,
      riskLevel: row.risk_level,
      riskScore: parseFloat(row.risk_score),
      contactInfo: typeof row.contact_info === 'string' ? JSON.parse(row.contact_info) : row.contact_info,
      businessInfo: typeof row.business_info === 'string' ? JSON.parse(row.business_info) : row.business_info,
      paymentInfo: typeof row.payment_info === 'string' ? JSON.parse(row.payment_info) : row.payment_info,
      statistics: typeof row.statistics === 'string' ? JSON.parse(row.statistics) : row.statistics,
      createdAt: row.created_at,
      updatedAt: row.updated_at
    };
  }

  private mapRowToChargeback(row: any): Chargeback {
    return {
      id: row.id,
      transactionId: row.transaction_id,
      merchantId: row.merchant_id,
      amount: parseFloat(row.amount),
      reason: row.reason,
      status: row.status,
      filedAt: row.filed_at,
      resolvedAt: row.resolved_at,
      evidence: typeof row.evidence === 'string' ? JSON.parse(row.evidence) : row.evidence
    };
  }

  private mapRowToDispute(row: any): Dispute {
    return {
      id: row.id,
      chargebackId: row.chargeback_id,
      merchantId: row.merchant_id,
      status: row.status,
      evidence: typeof row.evidence === 'string' ? JSON.parse(row.evidence) : row.evidence,
      notes: typeof row.notes === 'string' ? JSON.parse(row.notes) : row.notes,
      createdAt: row.created_at,
      updatedAt: row.updated_at
    };
  }
}
