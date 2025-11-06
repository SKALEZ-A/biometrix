import { Pool } from 'pg';

export interface FraudDetectionResult {
  id: string;
  transactionId: string;
  riskScore: number;
  fraudProbability: number;
  decision: 'approve' | 'decline' | 'review';
  modelVersion: string;
  features: Record<string, any>;
  reasons: string[];
  createdAt: Date;
}

export class FraudDetectionRepository {
  constructor(private pool: Pool) {}

  async saveFraudDetectionResult(result: Omit<FraudDetectionResult, 'id' | 'createdAt'>): Promise<FraudDetectionResult> {
    const query = `
      INSERT INTO fraud_detection_results (
        transaction_id, risk_score, fraud_probability, decision,
        model_version, features, reasons
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING *
    `;

    const values = [
      result.transactionId,
      result.riskScore,
      result.fraudProbability,
      result.decision,
      result.modelVersion,
      JSON.stringify(result.features),
      JSON.stringify(result.reasons)
    ];

    const dbResult = await this.pool.query(query, values);
    return this.mapRowToResult(dbResult.rows[0]);
  }

  async findByTransactionId(transactionId: string): Promise<FraudDetectionResult | null> {
    const query = 'SELECT * FROM fraud_detection_results WHERE transaction_id = $1 ORDER BY created_at DESC LIMIT 1';
    const result = await this.pool.query(query, [transactionId]);

    if (result.rows.length === 0) return null;
    return this.mapRowToResult(result.rows[0]);
  }

  async findHighRiskTransactions(limit: number = 100): Promise<FraudDetectionResult[]> {
    const query = `
      SELECT * FROM fraud_detection_results
      WHERE risk_score >= 70
      ORDER BY created_at DESC
      LIMIT $1
    `;
    const result = await this.pool.query(query, [limit]);
    return result.rows.map(this.mapRowToResult);
  }

  async getStatistics(startDate: Date, endDate: Date): Promise<{
    totalAnalyzed: number;
    fraudDetected: number;
    averageRiskScore: number;
    decisionBreakdown: Record<string, number>;
  }> {
    const query = `
      SELECT
        COUNT(*) as total_analyzed,
        SUM(CASE WHEN decision = 'decline' THEN 1 ELSE 0 END) as fraud_detected,
        AVG(risk_score) as average_risk_score,
        decision,
        COUNT(*) as decision_count
      FROM fraud_detection_results
      WHERE created_at BETWEEN $1 AND $2
      GROUP BY decision
    `;

    const result = await this.pool.query(query, [startDate, endDate]);

    const decisionBreakdown: Record<string, number> = {};
    let totalAnalyzed = 0;
    let fraudDetected = 0;
    let averageRiskScore = 0;

    result.rows.forEach(row => {
      decisionBreakdown[row.decision] = parseInt(row.decision_count);
      totalAnalyzed += parseInt(row.decision_count);
      if (row.decision === 'decline') {
        fraudDetected = parseInt(row.fraud_detected);
      }
      averageRiskScore = parseFloat(row.average_risk_score);
    });

    return {
      totalAnalyzed,
      fraudDetected,
      averageRiskScore,
      decisionBreakdown
    };
  }

  private mapRowToResult(row: any): FraudDetectionResult {
    return {
      id: row.id,
      transactionId: row.transaction_id,
      riskScore: parseFloat(row.risk_score),
      fraudProbability: parseFloat(row.fraud_probability),
      decision: row.decision,
      modelVersion: row.model_version,
      features: typeof row.features === 'string' ? JSON.parse(row.features) : row.features,
      reasons: typeof row.reasons === 'string' ? JSON.parse(row.reasons) : row.reasons,
      createdAt: row.created_at
    };
  }
}
