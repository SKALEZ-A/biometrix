import { Pool } from 'pg';
import { BiometricProfile, BiometricType } from '../models/biometric-profile.model';

export class BiometricRepository {
  constructor(private pool: Pool) {}

  async create(profile: Omit<BiometricProfile, 'id' | 'createdAt' | 'updatedAt'>): Promise<BiometricProfile> {
    const query = `
      INSERT INTO biometric_profiles (
        user_id, biometric_type, template_data, quality_score,
        enrollment_date, device_info, metadata
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING *
    `;

    const values = [
      profile.userId,
      profile.biometricType,
      profile.templateData,
      profile.qualityScore,
      profile.enrollmentDate,
      JSON.stringify(profile.deviceInfo),
      JSON.stringify(profile.metadata)
    ];

    const result = await this.pool.query(query, values);
    return this.mapRowToProfile(result.rows[0]);
  }

  async findByUserId(userId: string, biometricType?: BiometricType): Promise<BiometricProfile[]> {
    let query = 'SELECT * FROM biometric_profiles WHERE user_id = $1';
    const values: any[] = [userId];

    if (biometricType) {
      query += ' AND biometric_type = $2';
      values.push(biometricType);
    }

    query += ' ORDER BY created_at DESC';

    const result = await this.pool.query(query, values);
    return result.rows.map(this.mapRowToProfile);
  }

  async findById(profileId: string): Promise<BiometricProfile | null> {
    const query = 'SELECT * FROM biometric_profiles WHERE id = $1';
    const result = await this.pool.query(query, [profileId]);

    if (result.rows.length === 0) return null;
    return this.mapRowToProfile(result.rows[0]);
  }

  async update(profileId: string, updates: Partial<BiometricProfile>): Promise<BiometricProfile | null> {
    const setClauses: string[] = [];
    const values: any[] = [];
    let paramIndex = 1;

    if (updates.templateData !== undefined) {
      setClauses.push(`template_data = $${paramIndex++}`);
      values.push(updates.templateData);
    }

    if (updates.qualityScore !== undefined) {
      setClauses.push(`quality_score = $${paramIndex++}`);
      values.push(updates.qualityScore);
    }

    if (updates.metadata !== undefined) {
      setClauses.push(`metadata = $${paramIndex++}`);
      values.push(JSON.stringify(updates.metadata));
    }

    if (updates.isActive !== undefined) {
      setClauses.push(`is_active = $${paramIndex++}`);
      values.push(updates.isActive);
    }

    setClauses.push(`updated_at = $${paramIndex++}`);
    values.push(new Date());

    values.push(profileId);

    const query = `
      UPDATE biometric_profiles
      SET ${setClauses.join(', ')}
      WHERE id = $${paramIndex}
      RETURNING *
    `;

    const result = await this.pool.query(query, values);
    if (result.rows.length === 0) return null;
    return this.mapRowToProfile(result.rows[0]);
  }

  async delete(profileId: string): Promise<boolean> {
    const query = 'DELETE FROM biometric_profiles WHERE id = $1';
    const result = await this.pool.query(query, [profileId]);
    return result.rowCount > 0;
  }

  async softDelete(profileId: string): Promise<boolean> {
    const query = 'UPDATE biometric_profiles SET is_active = false, updated_at = $1 WHERE id = $2';
    const result = await this.pool.query(query, [new Date(), profileId]);
    return result.rowCount > 0;
  }

  async getStatistics(userId: string): Promise<{
    totalProfiles: number;
    byType: Record<BiometricType, number>;
    averageQuality: number;
  }> {
    const query = `
      SELECT
        COUNT(*) as total_profiles,
        biometric_type,
        COUNT(*) as type_count,
        AVG(quality_score) as avg_quality
      FROM biometric_profiles
      WHERE user_id = $1 AND is_active = true
      GROUP BY biometric_type
    `;

    const result = await this.pool.query(query, [userId]);

    const byType: Record<string, number> = {};
    let totalQuality = 0;
    let totalProfiles = 0;

    result.rows.forEach(row => {
      byType[row.biometric_type] = parseInt(row.type_count);
      totalQuality += parseFloat(row.avg_quality) * parseInt(row.type_count);
      totalProfiles += parseInt(row.type_count);
    });

    return {
      totalProfiles,
      byType: byType as Record<BiometricType, number>,
      averageQuality: totalProfiles > 0 ? totalQuality / totalProfiles : 0
    };
  }

  private mapRowToProfile(row: any): BiometricProfile {
    return {
      id: row.id,
      userId: row.user_id,
      biometricType: row.biometric_type,
      templateData: row.template_data,
      qualityScore: row.quality_score,
      enrollmentDate: row.enrollment_date,
      deviceInfo: typeof row.device_info === 'string' ? JSON.parse(row.device_info) : row.device_info,
      metadata: typeof row.metadata === 'string' ? JSON.parse(row.metadata) : row.metadata,
      isActive: row.is_active,
      createdAt: row.created_at,
      updatedAt: row.updated_at
    };
  }
}
