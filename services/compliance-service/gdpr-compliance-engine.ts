import { ConsentRecord, DataSubject, ProcessingActivity, AuditLog } from '../types/compliance-types';
import { DatabaseClient } from '../database/postgres-client';
import { Logger } from '../utils/logger';
import { CryptoService } from '../security/crypto-service';
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import moment from 'moment-timezone';

export interface ComplianceConfig {
  retentionPeriodDays: number;
  dataMinimizationRules: Map<string, string[]>;
  consentExpiryDays: number;
  piiFields: string[];
  auditLogRetentionDays: number;
}

export class GDPRComplianceEngine extends EventEmitter {
  private db: DatabaseClient;
  private logger: Logger;
  private crypto: CryptoService;
  private config: ComplianceConfig;
  private consents: Map<string, ConsentRecord> = new Map();
  private auditLogs: AuditLog[] = [];

  constructor(config: ComplianceConfig) {
    super();
    this.config = config;
    this.db = new DatabaseClient();
    this.logger = new Logger('GDPRComplianceEngine');
    this.crypto = new CryptoService();
    this.initialize();
  }

  private async initialize(): Promise<void> {
    await this.db.connect();
    await this.loadConsents();
    this.logger.info('GDPR Compliance Engine initialized');
  }

  private async loadConsents(): Promise<void> {
    const result = await this.db.query<ConsentRecord>('SELECT * FROM consents WHERE expiry > NOW()');
    result.rows.forEach(consent => {
      this.consents.set(consent.id, consent);
    });
    this.logger.info(`Loaded ${this.consents.size} active consents`);
  }

  // Consent Management (Article 6, 7 GDPR)
  async recordConsent(dataSubject: DataSubject, purposes: string[], legalBasis: 'consent' | 'contract' | 'legitimate_interest'): Promise<ConsentRecord> {
    const consentId = uuidv4();
    const record: ConsentRecord = {
      id: consentId,
      dataSubjectId: dataSubject.id,
      purposes: purposes,
      legalBasis: legalBasis,
      grantedAt: new Date().toISOString(),
      expiry: this.calculateExpiry(),
      version: '1.0',
      metadata: {
        ip: dataSubject.ip,
        userAgent: dataSubject.userAgent,
        consentText: `User ${dataSubject.id} consents to ${purposes.join(', ')} for ${legalBasis}`,
        signature: await this.crypto.signData(JSON.stringify({ purposes, legalBasis }))
      },
      status: 'active'
    };

    // Store encrypted
    const encryptedRecord = await this.crypto.encrypt(JSON.stringify(record));
    await this.db.query('INSERT INTO consents (id, encrypted_data, status) VALUES ($1, $2, $3)', 
      [consentId, encryptedRecord, 'active']);

    this.consents.set(consentId, record);
    this.emit('consent.granted', record);
    this.logAudit('consent_granted', { consentId, dataSubjectId: dataSubject.id, purposes });

    this.logger.info(`Recorded consent ${consentId} for ${dataSubject.id}`);
    return record;
  }

  async withdrawConsent(consentId: string, reason: string): Promise<void> {
    if (!this.consents.has(consentId)) {
      throw new Error(`Consent ${consentId} not found`);
    }

    const consent = this.consents.get(consentId)!;
    consent.status = 'withdrawn';
    consent.withdrawnAt = new Date().toISOString();
    consent.withdrawalReason = reason;

    const encryptedRecord = await this.crypto.encrypt(JSON.stringify(consent));
    await this.db.query('UPDATE consents SET encrypted_data = $1, status = $2, withdrawn_at = $3 WHERE id = $4',
      [encryptedRecord, 'withdrawn', new Date(), consentId]);

    this.consents.set(consentId, consent);
    this.emit('consent.withdrawn', consent);
    this.logAudit('consent_withdrawn', { consentId, reason });

    // Trigger data deletion if no other consents
    await this.purgePersonalData(consent.dataSubjectId);
    
    this.logger.info(`Consent ${consentId} withdrawn`);
  }

  private calculateExpiry(): string {
    return moment().add(this.config.consentExpiryDays, 'days').toISOString();
  }

  // Data Subject Rights (Articles 15-22 GDPR)
  async handleDataSubjectRequest(requestType: 'access' | 'rectification' | 'erasure' | 'restriction' | 'portability', 
                                dataSubjectId: string, details?: any): Promise<any> {
    this.logAudit(requestType, { dataSubjectId, details });

    switch (requestType) {
      case 'access':
        return await this.dataAccessRequest(dataSubjectId);
      
      case 'rectification':
        return await this.rectifyData(dataSubjectId, details);
      
      case 'erasure':
        return await this.eraseData(dataSubjectId, details?.reason);
      
      case 'restriction':
        return await this.restrictProcessing(dataSubjectId, details?.reason);
      
      case 'portability':
        return await this.dataPortabilityRequest(dataSubjectId, details?.format);
      
      default:
        throw new Error(`Unsupported request type: ${requestType}`);
    }
  }

  private async dataAccessRequest(dataSubjectId: string): Promise<Dict<string, any>> {
    // Verify identity first (biometric or other)
    await this.verifyDataSubject(dataSubjectId);

    const personalData = await this.db.query(`
      SELECT * FROM personal_data WHERE subject_id = $1 AND NOT is_deleted
    `, [dataSubjectId]);

    const consents = await this.db.query(`
      SELECT * FROM consents WHERE data_subject_id = $1
    `, [dataSubjectId]);

    const processingActivities = await this.getProcessingActivities(dataSubjectId);

    const response = {
      personalData: personalData.rows,
      consents: consents.rows,
      processingActivities: processingActivities,
      recipients: await this.getDataRecipients(dataSubjectId),
      retentionPeriod: this.config.retentionPeriodDays,
      lastUpdated: new Date().toISOString()
    };

    // Log the access request
    this.logAudit('data_access_request', { dataSubjectId, recordsFound: personalData.rowCount });

    // Anonymize after response if configured
    if (this.config.dataMinimizationRules.has('access_anonymize')) {
      await this.anonymizeAfterAccess(dataSubjectId);
    }

    return response;
  }

  private async rectifyData(dataSubjectId: string, corrections: Dict<string, any>): Promise<void> {
    await this.verifyDataSubject(dataSubjectId);

    // Validate corrections against PII fields
    const validFields = this.config.piiFields;
    const invalidFields = Object.keys(corrections).filter(field => !validFields.includes(field));
    if (invalidFields.length > 0) {
      throw new Error(`Invalid fields for rectification: ${invalidFields.join(', ')}`);
    }

    // Update data with audit trail
    for (const [field, value] of Object.entries(corrections)) {
      const hashedOldValue = await this.crypto.hashData(this.db.getField(dataSubjectId, field));
      await this.db.query(`
        UPDATE personal_data 
        SET ${field} = $1, last_modified = NOW(), modified_by = 'data_subject_rectification'
        WHERE subject_id = $2
      `, [value, dataSubjectId]);

      this.logAudit('data_rectification', { 
        dataSubjectId, 
        field, 
        oldValueHash: hashedOldValue,
        newValueHash: await this.crypto.hashData(value.toString())
      });
    }

    this.logger.info(`Data rectified for ${dataSubjectId}`);
  }

  async eraseData(dataSubjectId: string, reason?: string): Promise<{ deletedRecords: number }> {
    await this.verifyDataSubject(dataSubjectId);

    // Check for legal holds or obligations
    const legalHolds = await this.db.query('SELECT * FROM legal_holds WHERE subject_id = $1', [dataSubjectId]);
    if (legalHolds.rowCount > 0) {
      throw new Error('Cannot erase data due to active legal holds');
    }

    // Soft delete PII (GDPR Article 17)
    const deleted = await this.db.query(`
      UPDATE personal_data 
      SET is_deleted = true, deleted_at = NOW(), deletion_reason = $1, 
          deleted_by = 'data_subject_request'
      WHERE subject_id = $2 AND NOT is_deleted
      RETURNING id
    `, [reason || 'right_to_be_forgotten', dataSubjectId]);

    // Revoke all consents
    const consents = Array.from(this.consents.values()).filter(c => c.dataSubjectId === dataSubjectId);
    for (const consent of consents) {
      await this.withdrawConsent(consent.id, 'data_erasure_request');
    }

    // Purge from downstream systems
    await this.propagateErasure(dataSubjectId);

    this.emit('data.erasure', { dataSubjectId, reason, deletedCount: deleted.rowCount });
    this.logAudit('data_erasure', { dataSubjectId, reason, deletedRecords: deleted.rowCount });

    this.logger.info(`Data erasure completed for ${dataSubjectId}: ${deleted.rowCount} records`);
    return { deletedRecords: deleted.rowCount };
  }

  private async restrictProcessing(dataSubjectId: string, reason: string): Promise<void> {
    await this.db.query(`
      UPDATE personal_data 
      SET processing_restricted = true, restriction_reason = $1, restricted_at = NOW()
      WHERE subject_id = $2
    `, [reason, dataSubjectId]);

    // Notify processing activities
    const activities = await this.getProcessingActivities(dataSubjectId);
    activities.forEach(activity => {
      this.emit('processing.restricted', { ...activity, dataSubjectId, reason });
    });

    this.logAudit('processing_restriction', { dataSubjectId, reason });
    this.logger.info(`Processing restricted for ${dataSubjectId}`);
  }

  private async dataPortabilityRequest(dataSubjectId: string, format: 'json' | 'csv' = 'json'): Promise<string> {
    await this.verifyDataSubject(dataSubjectId);

    const data = await this.dataAccessRequest(dataSubjectId);
    
    const portableData = {
      dataSubject: { id: dataSubjectId },
      personalData: data.personalData,
      consents: data.consents,
      processingLog: data.processingActivities.slice(-100),  // Last 100 activities
      exportFormat: format,
      exportDate: new Date().toISOString(),
      controller: 'Biometric Fraud Prevention System',
      contact: 'privacy@company.com'
    };

    const filename = `data_export_${dataSubjectId}_${Date.now()}.${format}`;
    const filepath = `/tmp/${filename}`;

    if (format === 'json') {
      await this.writeFile(filepath, JSON.stringify(portableData, null, 2));
    } else {
      // Convert to CSV
      const csvData = this.convertToCSV(portableData);
      await this.writeFile(filepath, csvData);
    }

    // Secure download link (in production, use signed S3 URL)
    this.logAudit('data_portability', { dataSubjectId, format, recordsExported: data.personalData.length });

    return filepath;  // Return path or URL
  }

  // Data Protection Impact Assessment (DPIA) - Article 35
  async conductDPIA(processingActivity: ProcessingActivity): Promise<DPIAReport> {
    const risks = await this.assessRisks(processingActivity);
    const mitigations = this.recommendMitigations(risks);
    const safeguards = await this.implementSafeguards(processingActivity);

    const report: DPIAReport = {
      activityId: processingActivity.id,
      assessmentDate: new Date().toISOString(),
      dataController: 'Biometric Fraud System',
      processingDescription: processingActivity.description,
      dataSubjectsAffected: processingActivity.dataSubjectsAffected,
      highRisks: risks.filter(r => r.level === 'high'),
      mitigations: mitigations,
      safeguardsImplemented: safeguards,
      residualRisk: this.calculateResidualRisk(risks, mitigations),
      approvalStatus: 'pending',
      dpoReview: false,
      version: '1.0'
    };

    await this.db.query('INSERT INTO dpias (id, report) VALUES ($1, $2)', 
      [processingActivity.id, JSON.stringify(report)]);

    this.emit('dpia.completed', report);
    this.logAudit('dpia_conducted', { activityId: processingActivity.id, highRiskCount: report.highRisks.length });

    return report;
  }

  private async assessRisks(activity: ProcessingActivity): Promise<RiskAssessment[]> {
    const risks: RiskAssessment[] = [];

    // Systematic risk assessment
    if (activity.personalDataTypes.includes('biometric')) {
      risks.push({
        id: uuidv4(),
        category: 'data_sensitivity',
        description: 'Processing of biometric data (special category)',
        likelihood: 'high',
        impact: 'very_high',
        level: 'high',
        legalBasis: 'explicit_consent',
        mitigations: ['pseudonymization', 'access_controls']
      });
    }

    if (activity.dataVolume > 100000) {
      risks.push({
        id: uuidv4(),
        category: 'scale',
        description: 'Large-scale processing',
        likelihood: 'medium',
        impact: 'high',
        level: 'high',
        legalBasis: 'dpia_required',
        mitigations: ['data_minimization', 'retention_limits']
      });
    }

    // Automated decision-making risk
    if (activity.automatedDecisions && activity.profiling) {
      risks.push({
        id: uuidv4(),
        category: 'automated_decisions',
        description: 'Automated fraud decisions with profiling',
        likelihood: 'high',
        impact: 'high',
        level: 'high',
        legalBasis: 'article_22_exception',
        mitigations: ['human_review_threshold', 'explainability']
      });
    }

    // Third-party sharing
    if (activity.recipients.length > 0) {
      risks.push({
        id: uuidv4(),
        category: 'data_transfer',
        description: `Data sharing with ${activity.recipients.length} third parties`,
        likelihood: 'medium',
        impact: 'medium',
        level: 'medium',
        legalBasis: 'contractual',
        mitigations: ['dpas', 'scc', 'encryption']
      });
    }

    return risks;
  }

  private recommendMitigations(risks: RiskAssessment[]): Mitigation[] {
    const mitigations: Mitigation[] = [];

    risks.forEach(risk => {
      switch (risk.category) {
        case 'data_sensitivity':
          mitigations.push({
            riskId: risk.id,
            type: 'technical',
            description: 'Implement biometric data encryption at rest and in transit (AES-256)',
            implementation: 'crypto-service#encryptBiometric',
            effectiveness: 'high',
            cost: 'medium'
          });
          mitigations.push({
            riskId: risk.id,
            type: 'organizational',
            description: 'Require explicit consent with granular controls for biometric processing',
            implementation: 'consent-management#recordConsent',
            effectiveness: 'high',
            cost: 'low'
          });
          break;

        case 'automated_decisions':
          mitigations.push({
            riskId: risk.id,
            type: 'technical',
            description: 'Add human-in-the-loop review for high-risk scores (>0.8)',
            implementation: 'risk-engine#escalateReview',
            effectiveness: 'very_high',
            cost: 'medium'
          });
          mitigations.push({
            riskId: risk.id,
            type: 'technical',
            description: 'Implement model explainability using SHAP/LIME for fraud decisions',
            implementation: 'ml-service#explainPrediction',
            effectiveness: 'high',
            cost: 'high'
          });
          break;

        case 'scale':
          mitigations.push({
            riskId: risk.id,
            type: 'technical',
            description: 'Apply data minimization: retain only necessary biometric features',
            implementation: 'data-processor#applyMinimization',
            effectiveness: 'medium',
            cost: 'low'
          });
          break;

        // Add more case-specific mitigations
      }
    });

    return mitigations;
  }

  // Data Protection by Design/Default (Article 25)
  async enforceDataMinimization(data: any, purpose: string): Promise<any> {
    const rules = this.config.dataMinimizationRules.get(purpose) || [];
    const minimizedData = { ...data };

    this.config.piiFields.forEach(field => {
      if (data[field] && !rules.includes(field)) {
        // Pseudonymize or remove
        if (field.includes('biometric')) {
          minimizedData[field] = await this.crypto.pseudonymize(data[field]);
        } else {
          delete minimizedData[field];
        }
      }
    });

    this.logAudit('data_minimization', { purpose, fieldsRemoved: Object.keys(data).length - Object.keys(minimizedData).length });
    return minimizedData;
  }

  // Privacy-Enhancing Technologies
  async pseudonymizeBiometricData(biometricData: any): Promise<string> {
    const pseudonym = await this.crypto.hashData(JSON.stringify(biometricData), { salt: 'biometric_pseudonym' });
    return pseudonym.slice(0, 64);  // Truncated hash
  }

  async differentialPrivacyNoise(data: any, epsilon: number = 1.0): Promise<any> {
    // Add Laplace noise for numeric fields
    const noisyData = { ...data };
    for (const [key, value] of Object.entries(data)) {
      if (typeof value === 'number') {
        const sensitivity = 1.0;  // Assume unit sensitivity
        const noise = np.random.laplace(0, sensitivity / epsilon);
        noisyData[key] = value + noise;
      }
    }
    return noisyData;
  }

  // Audit and Logging (Accountability - Article 5(2))
  private logAudit(action: string, context: any): void {
    const log: AuditLog = {
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      action: action,
      actor: context.actor || 'system',
      dataSubjectId: context.dataSubjectId,
      resource: context.resource || action,
      details: JSON.stringify(context),
      ipAddress: context.ip,
      userAgent: context.userAgent,
      sensitivity: this.calculateLogSensitivity(context)
    };

    this.auditLogs.push(log);
    
    // Async store to DB
    this.db.query(`
      INSERT INTO audit_logs (id, action, actor, data_subject_id, details, ip_address, user_agent, sensitivity)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `, [
      log.id, action, log.actor, context.dataSubjectId, log.details,
      context.ip, context.userAgent, log.sensitivity
    ]).catch(err => this.logger.error(`Audit log failed: ${err}`));

    // Rotate logs if too many
    if (this.auditLogs.length > 10000) {
      this.rotateAuditLogs();
    }
  }

  private calculateLogSensitivity(context: any): 'low' | 'medium' | 'high' {
    if (context.dataSubjectId && ['erasure', 'access', 'rectification'].includes(context.action)) {
      return 'high';
    }
    if (context.piiFields || context.biometric) {
      return 'medium';
    }
    return 'low';
  }

  private async rotateAuditLogs(): Promise<void> {
    // Keep only last N days
    const cutoff = moment().subtract(this.config.auditLogRetentionDays, 'days').toDate();
    await this.db.query(`
      DELETE FROM audit_logs WHERE timestamp < $1 AND sensitivity = 'low'
    `, [cutoff]);

    // Anonymize old high-sensitivity logs
    await this.db.query(`
      UPDATE audit_logs 
      SET data_subject_id = NULL, details = anonymize_details(details)
      WHERE timestamp < $1 AND sensitivity = 'high'
    `, [cutoff]);

    this.logger.info('Audit logs rotated');
  }

  async getAuditTrail(dataSubjectId: string, fromDate?: string, toDate?: string): Promise<AuditLog[]> {
    const query = `
      SELECT * FROM audit_logs 
      WHERE data_subject_id = $1 
      ${fromDate ? 'AND timestamp >= $2' : ''} 
      ${toDate ? 'AND timestamp <= $3' : ''}
      ORDER BY timestamp DESC
    `;
    
    const params = [dataSubjectId];
    if (fromDate) params.push(fromDate);
    if (toDate) params.push(toDate);
    
    const result = await this.db.query<AuditLog>(query, params);
    
    // Filter sensitive info for export
    return result.rows.map(log => ({
      ...log,
      details: this.redactSensitiveDetails(log.details, dataSubjectId)
    }));
  }

  // Breach Detection and Notification (Articles 33, 34)
  async detectDataBreach(processingActivity: ProcessingActivity, threshold: number = 0.1): Promise<boolean> {
    // Monitor access patterns, anomalies
    const recentAccess = await this.getRecentAccess(processingActivity.id, 24);  // Last 24h
    const normalRate = await this.getBaselineAccessRate(processingActivity.id);
    
    const breachScore = Math.abs(recentAccess.length - normalRate) / normalRate;
    
    if (breachScore > threshold) {
      await this.notifyBreach(processingActivity, breachScore);
      return true;
    }
    
    return false;
  }

  private async getRecentAccess(activityId: string, hours: number): Promise<number> {
    const cutoff = moment().subtract(hours, 'hours').toISOString();
    const result = await this.db.query(`
      SELECT COUNT(*) as count FROM processing_logs 
      WHERE activity_id = $1 AND timestamp > $2
    `, [activityId, cutoff]);
    return parseInt(result.rows[0].count);
  }

  private async getBaselineAccessRate(activityId: string): Promise<number> {
    // 7-day average
    const result = await this.db.query(`
      SELECT AVG(daily_count) as avg FROM (
        SELECT COUNT(*) as daily_count FROM processing_logs 
        WHERE activity_id = $1 
        AND timestamp > NOW() - INTERVAL '7 days'
        GROUP BY DATE(timestamp)
      ) daily
    `, [activityId]);
    return parseFloat(result.rows[0].avg) || 0;
  }

  private async notifyBreach(activity: ProcessingActivity, breachScore: number): Promise<void> {
    const notification = {
      incidentId: uuidv4(),
      timestamp: new Date().toISOString(),
      activityId: activity.id,
      breachType: 'anomalous_access',
      severity: 'high',
      affectedSubjects: await this.calculateAffectedSubjects(activity),
      breachScore: breachScore,
      notificationSentTo: ['dpo@company.com', 'authorities@company.com'],
      description: `Anomalous access detected for ${activity.description}. Score: ${breachScore.toFixed(3)}`
    };

    // Send to DPO and authorities (within 72h)
    await this.sendBreachNotification(notification);
    
    // Log for supervisory authority
    await this.db.query('INSERT INTO breaches (incident_id, activity_id, severity, details) VALUES ($1, $2, $3, $4)',
      [notification.incidentId, activity.id, 'high', JSON.stringify(notification)]);

    this.emit('breach.detected', notification);
    this.logAudit('breach_detected', notification);

    this.logger.error(`Data breach detected: ${JSON.stringify(notification)}`);
  }

  private async calculateAffectedSubjects(activity: ProcessingActivity): Promise<number> {
    // Estimate based on data volume and breach scope
    const scope = activity.breachScope || 'limited';
    const multiplier = scope === 'widespread' ? 1 : scope === 'significant' ? 0.5 : 0.1;
    return Math.floor(activity.dataVolume * multiplier);
  }

  private async sendBreachNotification(notification: any): Promise<void> {
    // In production: email service, authorities API
    console.log('BREACH NOTIFICATION:', JSON.stringify(notification, null, 2));
    // await emailService.sendToDPO(notification);
    // if (affected > 250) await authorities.report(notification);
  }

  // Helper methods
  private async verifyDataSubject(dataSubjectId: string): Promise<boolean> {
    // Implement biometric or 2FA verification
    const verification = await this.crypto.verifyIdentity(dataSubjectId);
    if (!verification.valid) {
      throw new Error('Identity verification failed');
    }
    this.logAudit('identity_verification', { dataSubjectId, verified: true });
    return true;
  }

  private async getProcessingActivities(dataSubjectId: string): Promise<ProcessingActivity[]> {
    return await this.db.query(`
      SELECT * FROM processing_activities pa
      JOIN personal_data pd ON pa.data_category = pd.category
      WHERE pd.subject_id = $1
    `, [dataSubjectId]);
  }

  private async getDataRecipients(dataSubjectId: string): Promise<string[]> {
    const activities = await this.getProcessingActivities(dataSubjectId);
    return [...new Set(activities.flatMap(a => a.recipients || []))];
  }

  private async anonymizeAfterAccess(dataSubjectId: string): Promise<void> {
    // Pseudonymize accessed data
    await this.db.query(`
      UPDATE personal_data 
      SET is_anonymized = true, anonymized_at = NOW()
      WHERE subject_id = $1 AND last_accessed < NOW() - INTERVAL '30 days'
    `, [dataSubjectId]);
  }

  private async propagateErasure(dataSubjectId: string): Promise<void> {
    // Notify downstream processors
    const recipients = await this.getDataRecipients(dataSubjectId);
    for (const recipient of recipients) {
      await this.notifyRecipientErasure(recipient, dataSubjectId);
    }
  }

  private async notifyRecipientErasure(recipient: string, dataSubjectId: string): Promise<void> {
    // API call or message queue
    console.log(`Notifying ${recipient} to erase data for ${dataSubjectId}`);
  }

  private redactSensitiveDetails(details: string, dataSubjectId: string): string {
    // Remove PII from logs for export
    const redacted = JSON.parse(details);
    this.config.piiFields.forEach(field => {
      if (redacted[field]) {
        redacted[field] = '[REDACTED]';
      }
    });
    return JSON.stringify(redacted);
  }

  private async writeFile(filepath: string, content: string): Promise<void> {
    // Secure file write
    const encrypted = await this.crypto.encrypt(content);
    // In production: write to secure storage
    await this.db.storeSecureFile(filepath, encrypted);
  }

  private convertToCSV(data: any): string {
    // Simple CSV conversion for portability
    const headers = Object.keys(data.personalData[0] || {});
    let csv = headers.join(',') + '\n';
    data.personalData.forEach((row: any) => {
      csv += headers.map(h => `"${row[h] || ''}"`).join(',') + '\n';
    });
    return csv;
  }

  private calculateResidualRisk(risks: RiskAssessment[], mitigations: Mitigation[]): number {
    // Simple risk calculation: (likelihood * impact * remaining risks) / total possible
    let totalRisk = 0;
    let mitigatedRisk = 0;

    risks.forEach(risk => {
      const likelihoodScore = { low: 1, medium: 3, high: 5, very_high: 7 }[risk.likelihood] || 1;
      const impactScore = { low: 1, medium: 3, high: 5, very_high: 7 }[risk.impact] || 1;
      totalRisk += likelihoodScore * impactScore;

      // Check if mitigated
      const riskMitigations = mitigations.filter(m => m.riskId === risk.id);
      const mitigationEffectiveness = riskMitigations.reduce((sum, m) => 
        sum + ({ low: 0.2, medium: 0.5, high: 0.8, very_high: 1 }[m.effectiveness] || 0), 0
      ) / riskMitigations.length || 0;

      mitigatedRisk += (likelihoodScore * impactScore) * mitigationEffectiveness;
    });

    const residual = (totalRisk - mitigatedRisk) / totalRisk;
    return Math.max(0, Math.min(1, residual));
  }

  // Cleanup and expiry handling
  async cleanupExpiredData(): Promise<{ expiredConsents: number; purgedData: number }> {
    const expiredConsents = await this.db.query(`
      SELECT COUNT(*) as count FROM consents WHERE expiry < NOW() AND status = 'active'
    `);
    
    // Withdraw expired consents
    const expiredRecords = await this.db.query('SELECT id, data_subject_id FROM consents WHERE expiry < NOW() AND status = "active"');
    for (const record of expiredRecords.rows) {
      await this.withdrawConsent(record.id, 'consent_expiry');
    }

    const purgedData = await this.purgeExpiredData();

    this.logger.info(`Cleanup: ${expiredConsents.rows[0].count} expired consents, ${purgedData} data records purged`);
    return { expiredConsents: parseInt(expiredConsents.rows[0].count), purgedData };
  }

  private async purgeExpiredData(): Promise<number> {
    const cutoff = moment().subtract(this.config.retentionPeriodDays, 'days').toISOString();
    const result = await this.db.query(`
      DELETE FROM personal_data 
      WHERE created_at < $1 AND NOT is_deleted 
      RETURNING id
    `, [cutoff]);
    
    return result.rowCount;
  }

  // Getters for monitoring
  getActiveConsentsCount(): number {
    return this.consents.size;
  }

  getConsentRate(purpose: string): number {
    const purposeConsents = Array.from(this.consents.values()).filter(c => c.purposes.includes(purpose));
    return purposeConsents.length / this.numUsers * 100;  // Assuming numUsers available
  }

  async getComplianceMetrics(): Promise<ComplianceMetrics> {
    const totalConsents = await this.db.query('SELECT COUNT(*) FROM consents');
    const activeConsents = await this.db.query('SELECT COUNT(*) FROM consents WHERE status = "active"');
    const withdrawalRate = await this.db.query(`
      SELECT COUNT(*) FROM consents WHERE status = "withdrawn"
    `);
    const dpias = await this.db.query('SELECT COUNT(*) FROM dpias WHERE approval_status = "approved"');
    const breaches = await this.db.query('SELECT COUNT(*) FROM breaches WHERE timestamp > NOW() - INTERVAL "30 days"');

    return {
      totalConsents: parseInt(totalConsents.rows[0].count),
      activeConsents: parseInt(activeConsents.rows[0].count),
      consentRate: (parseInt(activeConsents.rows[0].count) / parseInt(totalConsents.rows[0].count)) * 100,
      withdrawalRate: parseInt(withdrawalRate.rows[0].count),
      dpiasCompleted: parseInt(dpias.rows[0].count),
      recentBreaches: parseInt(breaches.rows[0].count),
      dataRetentionCompliance: await this.checkRetentionCompliance(),
      lastAudit: await this.getLastAuditDate()
    };
  }

  private async checkRetentionCompliance(): Promise<boolean> {
    const overdue = await this.db.query(`
      SELECT COUNT(*) FROM personal_data 
      WHERE created_at < NOW() - INTERVAL '${this.config.retentionPeriodDays} days' 
      AND NOT is_deleted
    `);
    return parseInt(overdue.rows[0].count) === 0;
  }

  private async getLastAuditDate(): Promise<string> {
    const result = await this.db.query('SELECT MAX(timestamp) as last_audit FROM audit_logs');
    return result.rows[0].last_audit || 'never';
  }
}

// Types for type safety
export interface DPIAReport {
  activityId: string;
  assessmentDate: string;
  dataController: string;
  processingDescription: string;
  dataSubjectsAffected: number;
  highRisks: RiskAssessment[];
  mitigations: Mitigation[];
  safeguardsImplemented: Safeguard[];
  residualRisk: number;
  approvalStatus: 'pending' | 'approved' | 'rejected';
  dpoReview: boolean;
  version: string;
}

export interface RiskAssessment {
  id: string;
  category: string;
  description: string;
  likelihood: 'low' | 'medium' | 'high' | 'very_high';
  impact: 'low' | 'medium' | 'high' | 'very_high';
  level: 'low' | 'medium' | 'high';
  legalBasis: string;
  mitigations: string[];
}

export interface Mitigation {
  riskId: string;
  type: 'technical' | 'organizational' | 'procedural';
  description: string;
  implementation: string;
  effectiveness: 'low' | 'medium' | 'high' | 'very_high';
  cost: 'low' | 'medium' | 'high';
}

export interface Safeguard {
  id: string;
  type: string;
  description: string;
  status: 'implemented' | 'planned' | 'not_applicable';
}

export interface ComplianceMetrics {
  totalConsents: number;
  activeConsents: number;
  consentRate: number;
  withdrawalRate: number;
  dpiasCompleted: number;
  recentBreaches: number;
  dataRetentionCompliance: boolean;
  lastAudit: string;
}

// Usage example
async function initCompliance() {
  const config: ComplianceConfig = {
    retentionPeriodDays: 365,
    dataMinimizationRules: new Map([
      ['fraud_detection', ['name', 'email']],  // Retain only these
      ['analytics', ['user_id']]  // Pseudonymized
    ]),
    consentExpiryDays: 730,  // 2 years
    piiFields: ['name', 'email', 'phone', 'address', 'biometric_template', 'ssn'],
    auditLogRetentionDays: 1095  // 3 years for audits
  };

  const engine = new GDPRComplianceEngine(config);
  
  // Example consent
  const subject: DataSubject = { id: 'user_123', ip: '192.168.1.1', userAgent: 'Mozilla/5.0' };
  await engine.recordConsent(subject, ['fraud_prevention', 'analytics'], 'consent');
  
  // Example request
  const accessData = await engine.handleDataSubjectRequest('access', 'user_123');
  console.log('Access request result:', accessData);
  
  // DPIA
  const activity: ProcessingActivity = {
    id: 'fraud_biometric',
    description: 'Real-time biometric fraud detection',
    dataSubjectsAffected: 1000000,
    personalDataTypes: ['biometric', 'transaction'],
    automatedDecisions: true,
    profiling: true,
    recipients: ['ml-service', 'analytics-service'],
    dataVolume: 5000000,
    breachScope: 'limited'
  };
  
  const dpia = await engine.conductDPIA(activity);
  console.log('DPIA Report:', dpia);
}

if (require.main === module) {
  initCompliance().catch(console.error);
}

export default GDPRComplianceEngine;
