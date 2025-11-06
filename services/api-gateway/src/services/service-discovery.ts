import axios from 'axios';
import { logger } from '@shared/utils/logger';

interface ServiceInstance {
  id: string;
  name: string;
  host: string;
  port: number;
  healthy: boolean;
  lastHealthCheck: Date;
  metadata: Record<string, any>;
}

class ServiceDiscovery {
  private services: Map<string, ServiceInstance[]> = new Map();
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private readonly HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

  constructor() {
    this.initializeServices();
    this.startHealthChecks();
  }

  private initializeServices(): void {
    const serviceConfigs = [
      { name: 'biometric-service', host: process.env.BIOMETRIC_SERVICE_HOST || 'biometric-service', port: 3001 },
      { name: 'fraud-detection-service', host: process.env.FRAUD_DETECTION_SERVICE_HOST || 'fraud-detection-service', port: 3002 },
      { name: 'transaction-service', host: process.env.TRANSACTION_SERVICE_HOST || 'transaction-service', port: 3003 },
      { name: 'user-management-service', host: process.env.USER_MANAGEMENT_SERVICE_HOST || 'user-management-service', port: 3004 },
      { name: 'alert-service', host: process.env.ALERT_SERVICE_HOST || 'alert-service', port: 3005 },
      { name: 'compliance-service', host: process.env.COMPLIANCE_SERVICE_HOST || 'compliance-service', port: 3006 },
      { name: 'analytics-service', host: process.env.ANALYTICS_SERVICE_HOST || 'analytics-service', port: 3007 },
      { name: 'merchant-protection-service', host: process.env.MERCHANT_PROTECTION_SERVICE_HOST || 'merchant-protection-service', port: 3008 },
      { name: 'voice-service', host: process.env.VOICE_SERVICE_HOST || 'voice-service', port: 3009 },
      { name: 'webhook-service', host: process.env.WEBHOOK_SERVICE_HOST || 'webhook-service', port: 3010 },
      { name: 'notification-service', host: process.env.NOTIFICATION_SERVICE_HOST || 'notification-service', port: 3011 },
      { name: 'reporting-service', host: process.env.REPORTING_SERVICE_HOST || 'reporting-service', port: 3012 },
      { name: 'audit-service', host: process.env.AUDIT_SERVICE_HOST || 'audit-service', port: 3013 },
    ];

    serviceConfigs.forEach(config => {
      const instance: ServiceInstance = {
        id: `${config.name}-${config.host}-${config.port}`,
        name: config.name,
        host: config.host,
        port: config.port,
        healthy: true,
        lastHealthCheck: new Date(),
        metadata: {}
      };

      if (!this.services.has(config.name)) {
        this.services.set(config.name, []);
      }
      this.services.get(config.name)!.push(instance);
    });

    logger.info('Service discovery initialized', { serviceCount: this.services.size });
  }

  private startHealthChecks(): void {
    this.healthCheckInterval = setInterval(() => {
      this.performHealthChecks();
    }, this.HEALTH_CHECK_INTERVAL);
  }

  private async performHealthChecks(): Promise<void> {
    for (const [serviceName, instances] of this.services.entries()) {
      for (const instance of instances) {
        try {
          const response = await axios.get(`http://${instance.host}:${instance.port}/health`, {
            timeout: 5000
          });
          
          instance.healthy = response.status === 200;
          instance.lastHealthCheck = new Date();
          
          if (!instance.healthy) {
            logger.warn('Service health check failed', { serviceName, instanceId: instance.id });
          }
        } catch (error) {
          instance.healthy = false;
          instance.lastHealthCheck = new Date();
          logger.error('Service health check error', { serviceName, instanceId: instance.id, error });
        }
      }
    }
  }

  public getServiceInstance(serviceName: string): ServiceInstance | null {
    const instances = this.services.get(serviceName);
    if (!instances || instances.length === 0) {
      return null;
    }

    // Filter healthy instances
    const healthyInstances = instances.filter(i => i.healthy);
    if (healthyInstances.length === 0) {
      logger.warn('No healthy instances available', { serviceName });
      return instances[0]; // Return first instance even if unhealthy
    }

    // Round-robin load balancing
    const randomIndex = Math.floor(Math.random() * healthyInstances.length);
    return healthyInstances[randomIndex];
  }

  public getAllServices(): Map<string, ServiceInstance[]> {
    return this.services;
  }

  public registerService(serviceName: string, host: string, port: number, metadata?: Record<string, any>): void {
    const instance: ServiceInstance = {
      id: `${serviceName}-${host}-${port}`,
      name: serviceName,
      host,
      port,
      healthy: true,
      lastHealthCheck: new Date(),
      metadata: metadata || {}
    };

    if (!this.services.has(serviceName)) {
      this.services.set(serviceName, []);
    }

    const instances = this.services.get(serviceName)!;
    const existingIndex = instances.findIndex(i => i.id === instance.id);
    
    if (existingIndex >= 0) {
      instances[existingIndex] = instance;
    } else {
      instances.push(instance);
    }

    logger.info('Service registered', { serviceName, instanceId: instance.id });
  }

  public deregisterService(serviceName: string, instanceId: string): void {
    const instances = this.services.get(serviceName);
    if (instances) {
      const filteredInstances = instances.filter(i => i.id !== instanceId);
      this.services.set(serviceName, filteredInstances);
      logger.info('Service deregistered', { serviceName, instanceId });
    }
  }

  public shutdown(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
    logger.info('Service discovery shutdown');
  }
}

export const serviceDiscovery = new ServiceDiscovery();
