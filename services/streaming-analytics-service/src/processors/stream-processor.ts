import { EventEmitter } from 'events';
import { KafkaClient } from '../../../packages/shared/src/queue/kafka-client';
import { RedisClient } from '../../../packages/shared/src/cache/redis';

export interface StreamEvent {
  id: string;
  timestamp: number;
  eventType: string;
  payload: any;
  metadata: Record<string, any>;
}

export interface ProcessingResult {
  processedEvents: number;
  failedEvents: number;
  latency: number;
  throughput: number;
}

export class StreamProcessor extends EventEmitter {
  private kafkaClient: KafkaClient;
  private redisClient: RedisClient;
  private processingBuffer: Map<string, StreamEvent[]>;
  private checkpointInterval: number = 5000;
  private maxBufferSize: number = 10000;

  constructor() {
    super();
    this.kafkaClient = new KafkaClient();
    this.redisClient = new RedisClient();
    this.processingBuffer = new Map();
    this.initializeCheckpointing();
  }

  async process(event: StreamEvent): Promise<ProcessingResult> {
    const startTime = Date.now();
    
    try {
      await this.validateEvent(event);
      await this.enrichEvent(event);
      await this.filterEvent(event);
      await this.transformEvent(event);
      await this.routeEvent(event);
      
      const latency = Date.now() - startTime;
      
      return {
        processedEvents: 1,
        failedEvents: 0,
        latency,
        throughput: 1000 / latency
      };
    } catch (error) {
      console.error('Stream processing error:', error);
      return {
        processedEvents: 0,
        failedEvents: 1,
        latency: Date.now() - startTime,
        throughput: 0
      };
    }
  }

  async processBatch(events: StreamEvent[]): Promise<ProcessingResult> {
    const startTime = Date.now();
    let processed = 0;
    let failed = 0;

    for (const event of events) {
      try {
        await this.process(event);
        processed++;
      } catch (error) {
        failed++;
      }
    }

    const latency = Date.now() - startTime;
    
    return {
      processedEvents: processed,
      failedEvents: failed,
      latency,
      throughput: (processed * 1000) / latency
    };
  }

  private async validateEvent(event: StreamEvent): Promise<void> {
    if (!event.id || !event.timestamp || !event.eventType) {
      throw new Error('Invalid event structure');
    }

    if (event.timestamp > Date.now() + 60000) {
      throw new Error('Event timestamp is too far in the future');
    }

    if (event.timestamp < Date.now() - 86400000) {
      throw new Error('Event is too old');
    }
  }

  private async enrichEvent(event: StreamEvent): Promise<void> {
    const cachedData = await this.redisClient.get(`enrichment:${event.eventType}`);
    
    if (cachedData) {
      event.metadata = {
        ...event.metadata,
        ...JSON.parse(cachedData)
      };
    }

    event.metadata.processingTimestamp = Date.now();
    event.metadata.processorId = process.pid;
  }

  private async filterEvent(event: StreamEvent): Promise<void> {
    const filterRules = await this.getFilterRules(event.eventType);
    
    for (const rule of filterRules) {
      if (!this.evaluateFilterRule(event, rule)) {
        throw new Error(`Event filtered by rule: ${rule.name}`);
      }
    }
  }

  private async transformEvent(event: StreamEvent): Promise<void> {
    const transformations = await this.getTransformations(event.eventType);
    
    for (const transformation of transformations) {
      event.payload = await this.applyTransformation(event.payload, transformation);
    }
  }

  private async routeEvent(event: StreamEvent): Promise<void> {
    const routingKey = this.determineRoutingKey(event);
    await this.kafkaClient.publish(routingKey, event);
    
    this.addToBuffer(routingKey, event);
  }

  private determineRoutingKey(event: StreamEvent): string {
    const typeMapping: Record<string, string> = {
      'transaction': 'fraud-detection-events',
      'biometric': 'biometric-verification-events',
      'alert': 'alert-notification-events',
      'audit': 'audit-log-events'
    };

    return typeMapping[event.eventType] || 'default-events';
  }

  private addToBuffer(key: string, event: StreamEvent): void {
    if (!this.processingBuffer.has(key)) {
      this.processingBuffer.set(key, []);
    }

    const buffer = this.processingBuffer.get(key)!;
    buffer.push(event);

    if (buffer.length >= this.maxBufferSize) {
      this.flushBuffer(key);
    }
  }

  private async flushBuffer(key: string): Promise<void> {
    const buffer = this.processingBuffer.get(key);
    
    if (buffer && buffer.length > 0) {
      await this.persistCheckpoint(key, buffer);
      this.processingBuffer.set(key, []);
    }
  }

  private async persistCheckpoint(key: string, events: StreamEvent[]): Promise<void> {
    const checkpoint = {
      key,
      lastEventId: events[events.length - 1].id,
      lastTimestamp: events[events.length - 1].timestamp,
      eventCount: events.length,
      checkpointTime: Date.now()
    };

    await this.redisClient.set(
      `checkpoint:${key}`,
      JSON.stringify(checkpoint),
      3600
    );
  }

  private initializeCheckpointing(): void {
    setInterval(() => {
      for (const key of this.processingBuffer.keys()) {
        this.flushBuffer(key);
      }
    }, this.checkpointInterval);
  }

  private async getFilterRules(eventType: string): Promise<any[]> {
    return [
      { name: 'not-null', field: 'payload', operator: 'exists' },
      { name: 'valid-timestamp', field: 'timestamp', operator: 'range', min: 0, max: Date.now() }
    ];
  }

  private evaluateFilterRule(event: StreamEvent, rule: any): boolean {
    switch (rule.operator) {
      case 'exists':
        return event[rule.field] !== null && event[rule.field] !== undefined;
      case 'range':
        return event[rule.field] >= rule.min && event[rule.field] <= rule.max;
      default:
        return true;
    }
  }

  private async getTransformations(eventType: string): Promise<any[]> {
    return [
      { type: 'normalize', fields: ['payload'] },
      { type: 'sanitize', fields: ['metadata'] }
    ];
  }

  private async applyTransformation(data: any, transformation: any): Promise<any> {
    switch (transformation.type) {
      case 'normalize':
        return this.normalizeData(data);
      case 'sanitize':
        return this.sanitizeData(data);
      default:
        return data;
    }
  }

  private normalizeData(data: any): any {
    if (typeof data === 'string') {
      return data.trim().toLowerCase();
    }
    return data;
  }

  private sanitizeData(data: any): any {
    if (typeof data === 'object') {
      const sanitized: any = {};
      for (const key in data) {
        if (!key.startsWith('_')) {
          sanitized[key] = data[key];
        }
      }
      return sanitized;
    }
    return data;
  }
}
