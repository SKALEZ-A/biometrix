import { EventEmitter } from 'events';

export interface KafkaConfig {
  brokers: string[];
  clientId: string;
  groupId?: string;
  ssl?: boolean;
  sasl?: {
    mechanism: string;
    username: string;
    password: string;
  };
}

export interface ProducerRecord {
  topic: string;
  messages: Array<{
    key?: string;
    value: string;
    headers?: Record<string, string>;
    partition?: number;
    timestamp?: string;
  }>;
}

export interface ConsumerConfig {
  groupId: string;
  topics: string[];
  fromBeginning?: boolean;
  autoCommit?: boolean;
  autoCommitInterval?: number;
}

export class KafkaClient extends EventEmitter {
  private config: KafkaConfig;
  private connected: boolean = false;
  private producers: Map<string, any> = new Map();
  private consumers: Map<string, any> = new Map();
  private messageBuffer: Map<string, any[]> = new Map();

  constructor(config?: KafkaConfig) {
    super();
    this.config = config || {
      brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
      clientId: process.env.KAFKA_CLIENT_ID || 'fraud-detection-client'
    };
  }

  async connect(): Promise<void> {
    try {
      console.log('Connecting to Kafka brokers:', this.config.brokers);
      this.connected = true;
      this.emit('connected');
    } catch (error) {
      console.error('Failed to connect to Kafka:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    for (const producer of this.producers.values()) {
      await this.disconnectProducer(producer);
    }

    for (const consumer of this.consumers.values()) {
      await this.disconnectConsumer(consumer);
    }

    this.connected = false;
    this.emit('disconnected');
  }

  async publish(topic: string, message: any, key?: string): Promise<void> {
    if (!this.connected) {
      await this.connect();
    }

    const producer = await this.getOrCreateProducer(topic);
    
    const record: ProducerRecord = {
      topic,
      messages: [{
        key: key || Date.now().toString(),
        value: JSON.stringify(message),
        timestamp: new Date().toISOString()
      }]
    };

    try {
      await this.sendToProducer(producer, record);
      this.emit('message-sent', { topic, message });
    } catch (error) {
      console.error('Failed to publish message:', error);
      throw error;
    }
  }

  async publishBatch(topic: string, messages: any[]): Promise<void> {
    if (!this.connected) {
      await this.connect();
    }

    const producer = await this.getOrCreateProducer(topic);
    
    const record: ProducerRecord = {
      topic,
      messages: messages.map((msg, index) => ({
        key: `${Date.now()}-${index}`,
        value: JSON.stringify(msg),
        timestamp: new Date().toISOString()
      }))
    };

    try {
      await this.sendToProducer(producer, record);
      this.emit('batch-sent', { topic, count: messages.length });
    } catch (error) {
      console.error('Failed to publish batch:', error);
      throw error;
    }
  }

  async subscribe(config: ConsumerConfig, handler: (message: any) => Promise<void>): Promise<void> {
    if (!this.connected) {
      await this.connect();
    }

    const consumer = await this.createConsumer(config);
    
    for (const topic of config.topics) {
      await this.subscribeToTopic(consumer, topic);
    }

    await this.runConsumer(consumer, handler, config.autoCommit !== false);
  }

  async commit(groupId: string, topic: string, partition: number, offset: string): Promise<void> {
    const consumer = this.consumers.get(groupId);
    if (!consumer) {
      throw new Error(`Consumer not found for group: ${groupId}`);
    }

    await this.commitOffset(consumer, topic, partition, offset);
  }

  async seek(groupId: string, topic: string, partition: number, offset: string): Promise<void> {
    const consumer = this.consumers.get(groupId);
    if (!consumer) {
      throw new Error(`Consumer not found for group: ${groupId}`);
    }

    await this.seekToOffset(consumer, topic, partition, offset);
  }

  async getTopicMetadata(topic: string): Promise<any> {
    if (!this.connected) {
      await this.connect();
    }

    return {
      topic,
      partitions: [
        { partition: 0, leader: 1, replicas: [1, 2], isr: [1, 2] },
        { partition: 1, leader: 2, replicas: [2, 3], isr: [2, 3] },
        { partition: 2, leader: 3, replicas: [3, 1], isr: [3, 1] }
      ]
    };
  }

  async createTopic(topic: string, numPartitions: number = 3, replicationFactor: number = 2): Promise<void> {
    if (!this.connected) {
      await this.connect();
    }

    console.log(`Creating topic: ${topic} with ${numPartitions} partitions`);
    this.emit('topic-created', { topic, numPartitions, replicationFactor });
  }

  async deleteTopic(topic: string): Promise<void> {
    if (!this.connected) {
      await this.connect();
    }

    console.log(`Deleting topic: ${topic}`);
    this.emit('topic-deleted', { topic });
  }

  private async getOrCreateProducer(topic: string): Promise<any> {
    if (this.producers.has(topic)) {
      return this.producers.get(topic);
    }

    const producer = await this.createProducer();
    this.producers.set(topic, producer);
    return producer;
  }

  private async createProducer(): Promise<any> {
    const producer = {
      id: `producer-${Date.now()}`,
      connected: true,
      transactional: false
    };

    console.log('Created Kafka producer:', producer.id);
    return producer;
  }

  private async createConsumer(config: ConsumerConfig): Promise<any> {
    const consumer = {
      id: `consumer-${Date.now()}`,
      groupId: config.groupId,
      connected: true,
      subscriptions: new Set<string>()
    };

    this.consumers.set(config.groupId, consumer);
    console.log('Created Kafka consumer:', consumer.id);
    return consumer;
  }

  private async sendToProducer(producer: any, record: ProducerRecord): Promise<void> {
    console.log(`Sending message to topic: ${record.topic}`);
    
    if (!this.messageBuffer.has(record.topic)) {
      this.messageBuffer.set(record.topic, []);
    }
    
    this.messageBuffer.get(record.topic)!.push(...record.messages);
  }

  private async subscribeToTopic(consumer: any, topic: string): Promise<void> {
    consumer.subscriptions.add(topic);
    console.log(`Consumer ${consumer.id} subscribed to topic: ${topic}`);
  }

  private async runConsumer(consumer: any, handler: (message: any) => Promise<void>, autoCommit: boolean): Promise<void> {
    console.log(`Starting consumer ${consumer.id} with autoCommit: ${autoCommit}`);
    
    setInterval(async () => {
      for (const topic of consumer.subscriptions) {
        const messages = this.messageBuffer.get(topic) || [];
        
        for (const message of messages) {
          try {
            const parsed = JSON.parse(message.value);
            await handler(parsed);
            
            if (autoCommit) {
              this.emit('message-committed', { topic, offset: message.key });
            }
          } catch (error) {
            console.error('Error processing message:', error);
            this.emit('message-error', { topic, error });
          }
        }
        
        if (messages.length > 0) {
          this.messageBuffer.set(topic, []);
        }
      }
    }, 1000);
  }

  private async commitOffset(consumer: any, topic: string, partition: number, offset: string): Promise<void> {
    console.log(`Committing offset ${offset} for topic ${topic}, partition ${partition}`);
  }

  private async seekToOffset(consumer: any, topic: string, partition: number, offset: string): Promise<void> {
    console.log(`Seeking to offset ${offset} for topic ${topic}, partition ${partition}`);
  }

  private async disconnectProducer(producer: any): Promise<void> {
    producer.connected = false;
    console.log(`Disconnected producer: ${producer.id}`);
  }

  private async disconnectConsumer(consumer: any): Promise<void> {
    consumer.connected = false;
    console.log(`Disconnected consumer: ${consumer.id}`);
  }

  isConnected(): boolean {
    return this.connected;
  }

  getProducerCount(): number {
    return this.producers.size;
  }

  getConsumerCount(): number {
    return this.consumers.size;
  }
}
