import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { AlertOrchestratorService } from '../src/services/alert-orchestrator.service';
import { AlertPriorityService } from '../src/services/alert-priority.service';

describe('AlertOrchestratorService', () => {
  let alertOrchestrator: AlertOrchestratorService;
  let alertPriority: AlertPriorityService;

  beforeEach(() => {
    alertPriority = new AlertPriorityService();
    alertOrchestrator = new AlertOrchestratorService(alertPriority);
  });

  describe('createAlert', () => {
    it('should create a fraud alert with correct priority', async () => {
      const alertData = {
        type: 'fraud' as const,
        severity: 'high' as const,
        title: 'Suspicious Transaction Detected',
        description: 'Multiple high-value transactions from new location',
        source: 'fraud-detection-service',
        metadata: {
          transactionId: 'txn-123',
          userId: 'user-456',
          amount: 5000
        }
      };

      const alert = await alertOrchestrator.createAlert(alertData);

      expect(alert).toBeDefined();
      expect(alert.type).toBe('fraud');
      expect(alert.severity).toBe('high');
      expect(alert.status).toBe('open');
    });

    it('should assign correct priority based on severity', () => {
      const criticalPriority = alertPriority.calculatePriority('critical', 'fraud');
      const highPriority = alertPriority.calculatePriority('high', 'fraud');
      const mediumPriority = alertPriority.calculatePriority('medium', 'fraud');

      expect(criticalPriority).toBeGreaterThan(highPriority);
      expect(highPriority).toBeGreaterThan(mediumPriority);
    });
  });

  describe('acknowledgeAlert', () => {
    it('should update alert status to acknowledged', async () => {
      const alert = await alertOrchestrator.createAlert({
        type: 'fraud',
        severity: 'medium',
        title: 'Test Alert',
        description: 'Test Description',
        source: 'test'
      });

      const acknowledged = await alertOrchestrator.acknowledgeAlert(
        alert.id,
        'user-123'
      );

      expect(acknowledged.status).toBe('acknowledged');
      expect(acknowledged.acknowledgedBy).toBe('user-123');
    });
  });
});
