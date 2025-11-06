import { Pool } from 'pg';

export interface Voiceprint {
  id: string;
  userId: string;
  voiceprintData: Buffer;
  qualityScore: number;
  enrollmentDate: Date;
  deviceInfo: Record<string, any>;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export class VoiceRepository {
  constructor(private pool: Pool) {}

  async createVoiceprint(voiceprint: Omit<Voiceprint, 'id' | 'createdAt' | 'updatedAt'>): Promise<Voiceprint> {
    const query = `
      INSERT INTO voiceprints (
        user_id, voiceprint_data, quality_score, enrollment_date,
        device_info, is_active
      ) VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING *
    `;

    const values = [
      voiceprint.userId,
      voiceprint.voiceprintData,
      voiceprint.qualityScore,
      voiceprint.enrollmentDate,
      JSON.stringify(voiceprint.deviceInfo),
      voiceprint.isActive
    ];

    const result = await this.pool.query(query, values);
    return this.mapRowToVoiceprint(result.rows[0]);
  }

  async findByUserId(userId: string): Promise<Voiceprint | null> {
    const query = 'SELECT * FROM voiceprints WHERE user_id = $1 AND is_active = true ORDER BY created_at DESC LIMIT 1';
    const result = await this.pool.query(query, [userId]);

    if (result.rows.length === 0) return null;
    return this.mapRowToVoiceprint(result.rows[0]);
  }

  async update(voiceprintId: string, updates: Partial<Voiceprint>): Promise<Voiceprint | null> {
    const setClauses: string[] = [];
    const values: any[] = [];
    let paramIndex = 1;

    if (updates.voiceprintData !== undefined) {
      setClauses.push(`voiceprint_data = $${paramIndex++}`);
      values.push(updates.voiceprintData);
    }

    if (updates.qualityScore !== undefined) {
      setClauses.push(`quality_score = $${paramIndex++}`);
      values.push(updates.qualityScore);
    }

    if (updates.isActive !== undefined) {
      setClauses.push(`is_active = $${paramIndex++}`);
      values.push(updates.isActive);
    }

    setClauses.push(`updated_at = $${paramIndex++}`);
    values.push(new Date());

    values.push(voiceprintId);

    const query = `
      UPDATE voiceprints
      SET ${setClauses.join(', ')}
      WHERE id = $${paramIndex}
      RETURNING *
    `;

    const result = await this.pool.query(query, values);
    if (result.rows.length === 0) return null;
    return this.mapRowToVoiceprint(result.rows[0]);
  }

  async delete(voiceprintId: string): Promise<boolean> {
    const query = 'UPDATE voiceprints SET is_active = false, updated_at = $1 WHERE id = $2';
    const result = await this.pool.query(query, [new Date(), voiceprintId]);
    return result.rowCount > 0;
  }

  private mapRowToVoiceprint(row: any): Voiceprint {
    return {
      id: row.id,
      userId: row.user_id,
      voiceprintData: row.voiceprint_data,
      qualityScore: parseFloat(row.quality_score),
      enrollmentDate: row.enrollment_date,
      deviceInfo: typeof row.device_info === 'string' ? JSON.parse(row.device_info) : row.device_info,
      isActive: row.is_active,
      createdAt: row.created_at,
      updatedAt: row.updated_at
    };
  }
}
