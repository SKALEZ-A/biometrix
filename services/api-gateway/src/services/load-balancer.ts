import { serviceDiscovery } from './service-discovery';
import { logger } from '@shared/utils/logger';

export enum LoadBalancingStrategy {
  ROUND_ROBIN = 'round_robin',
  LEAST_CONNECTIONS = 'least_connections',
  RANDOM = 'random',
  WEIGHTED = 'weighted'
}

interface ServiceMetrics {
  activeConnections: number;
  totalRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  weight: number;
}

class LoadBalancer {
  private serviceMetrics: Map<string, ServiceMetrics> = new Map();
  private roundRobinCounters: Map<string, number> = new Map();
  private strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN;

  constructor(strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) {
    this.strategy = strategy;
  }

  public setStrategy(strategy: LoadBalancingStrategy): void {
    this.strategy = strategy;
    logger.info('Load balancing strategy changed', { strategy });
  }

  public getServiceUrl(serviceName: string): string | null {
    const instance = this.selectInstance(serviceName);
    if (!instance) {
      logger.error('No service instance available', { serviceName });
      return null;
    }

    return `http://${instance.host}:${instance.port}`;
  }

  private selectInstance(serviceName: string) {
    const instances = serviceDiscovery.getAllServices().get(serviceName);
    if (!instances || instances.length === 0) {
      return null;
    }

    const healthyInstances = instances.filter(i => i.healthy);
    if (healthyInstances.length === 0) {
      return null;
    }

    switch (this.strategy) {
      case LoadBalancingStrategy.ROUND_ROBIN:
        return this.roundRobinSelect(serviceName, healthyInstances);
      
      case LoadBalancingStrategy.LEAST_CONNECTIONS:
        return this.leastConnectionsSelect(healthyInstances);
      
      case LoadBalancingStrategy.RANDOM:
        return this.randomSelect(healthyInstances);
      
      case LoadBalancingStrategy.WEIGHTED:
        return this.weightedSelect(healthyInstances);
      
      default:
        return this.roundRobinSelect(serviceName, healthyInstances);
    }
  }

  private roundRobinSelect(serviceName: string, instances: any[]): any {
    if (!this.roundRobinCounters.has(serviceName)) {
      this.roundRobinCounters.set(serviceName, 0);
    }

    const counter = this.roundRobinCounters.get(serviceName)!;
    const instance = instances[counter % instances.length];
    this.roundRobinCounters.set(serviceName, counter + 1);

    return instance;
  }

  private leastConnectionsSelect(instances: any[]): any {
    let selectedInstance = instances[0];
    let minConnections = this.getMetrics(selectedInstance.id).activeConnections;

    for (const instance of instances) {
      const connections = this.getMetrics(instance.id).activeConnections;
      if (connections < minConnections) {
        minConnections = connections;
        selectedInstance = instance;
      }
    }

    return selectedInstance;
  }

  private randomSelect(instances: any[]): any {
    const randomIndex = Math.floor(Math.random() * instances.length);
    return instances[randomIndex];
  }

  private weightedSelect(instances: any[]): any {
    const totalWeight = instances.reduce((sum, instance) => {
      return sum + this.getMetrics(instance.id).weight;
    }, 0);

    let random = Math.random() * totalWeight;
    
    for (const instance of instances) {
      const weight = this.getMetrics(instance.id).weight;
      random -= weight;
      if (random <= 0) {
        return instance;
      }
    }

    return instances[0];
  }

  private getMetrics(instanceId: string): ServiceMetrics {
    if (!this.serviceMetrics.has(instanceId)) {
      this.serviceMetrics.set(instanceId, {
        activeConnections: 0,
        totalRequests: 0,
        failedRequests: 0,
        averageResponseTime: 0,
        weight: 1
      });
    }
    return this.serviceMetrics.get(instanceId)!;
  }

  public incrementConnections(instanceId: string): void {
    const metrics = this.getMetrics(instanceId);
    metrics.activeConnections++;
    metrics.totalRequests++;
  }

  public decrementConnections(instanceId: string): void {
    const metrics = this.getMetrics(instanceId);
    metrics.activeConnections = Math.max(0, metrics.activeConnections - 1);
  }

  public recordFailure(instanceId: string): void {
    const metrics = this.getMetrics(instanceId);
    metrics.failedRequests++;
  }

  public updateResponseTime(instanceId: string, responseTime: number): void {
    const metrics = this.getMetrics(instanceId);
    metrics.averageResponseTime = 
      (metrics.averageResponseTime * (metrics.totalRequests - 1) + responseTime) / metrics.totalRequests;
  }

  public setWeight(instanceId: string, weight: number): void {
    const metrics = this.getMetrics(instanceId);
    metrics.weight = weight;
  }

  public getMetricsSnapshot(): Map<string, ServiceMetrics> {
    return new Map(this.serviceMetrics);
  }
}

export const loadBalancer = new LoadBalancer();
