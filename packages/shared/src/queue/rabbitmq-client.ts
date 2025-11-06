import amqp, { Connection, Channel, ConsumeMessage } from 'amqplib';
import { logger } from '../utils/logger';

interface RabbitMQConfig {
  url: string;
  prefetchCount?: number;
  reconnectDelay?: number;
}

export class RabbitMQClient {
  private connection: Connection | null = null;
  private channel: Channel | null = null;
  private config: RabbitMQConfig;
  private reconnecting: boolean = false;

  constructor(config: RabbitMQConfig) {
    this.config = {
      prefetchCount: 10,
      reconnectDelay: 5000,
      ...config
    };
  }

  async connect(): Promise<void> {
    try {
      this.connection = await amqp.connect(this.config.url);
      this.channel = await this.connection.createChannel();
      await this.channel.prefetch(this.config.prefetchCount!);

      this.connection.on('error', (err) => {
        logger.error('RabbitMQ connection error', { error: err });
        this.reconnect();
      });

      this.connection.on('close', () => {
        logger.warn('RabbitMQ connection closed');
        this.reconnect();
      });

      logger.info('RabbitMQ connected successfully');
    } catch (error) {
      logger.error('RabbitMQ connection failed', { error });
      throw error;
    }
  }

  private async reconnect(): Promise<void> {
    if (this.reconnecting) return;
    
    this.reconnecting = true;
    logger.info('Attempting to reconnect to RabbitMQ');

    setTimeout(async () => {
      try {
        await this.connect();
        this.reconnecting = false;
      } catch (error) {
        logger.error('Reconnection failed', { error });
        this.reconnecting = false;
        this.reconnect();
      }
    }, this.config.reconnectDelay);
  }

  async assertQueue(queueName: string, options?: any): Promise<void> {
    if (!this.channel) throw new Error('Channel not initialized');
    await this.channel.assertQueue(queueName, options);
  }

  async assertExchange(exchangeName: string, type: string, options?: any): Promise<void> {
    if (!this.channel) throw new Error('Channel not initialized');
    await this.channel.assertExchange(exchangeName, type, options);
  }

  async bindQueue(queue: string, exchange: string, routingKey: string): Promise<void> {
    if (!this.channel) throw new Error('Channel not initialized');
    await this.channel.bindQueue(queue, exchange, routingKey);
  }

  async publish(exchange: string, routingKey: string, content: any): Promise<boolean> {
    if (!this.channel) throw new Error('Channel not initialized');
    
    const message = Buffer.from(JSON.stringify(content));
    return this.channel.publish(exchange, routingKey, message, { persistent: true });
  }

  async sendToQueue(queue: string, content: any): Promise<boolean> {
    if (!this.channel) throw new Error('Channel not initialized');
    
    const message = Buffer.from(JSON.stringify(content));
    return this.channel.sendToQueue(queue, message, { persistent: true });
  }

  async consume(
    queue: string,
    onMessage: (msg: ConsumeMessage | null) => Promise<void>,
    options?: any
  ): Promise<void> {
    if (!this.channel) throw new Error('Channel not initialized');
    
    await this.channel.consume(queue, async (msg) => {
      try {
        await onMessage(msg);
        if (msg) {
          this.channel!.ack(msg);
        }
      } catch (error) {
        logger.error('Message processing failed', { error });
        if (msg) {
          this.channel!.nack(msg, false, true);
        }
      }
    }, options);
  }

  async disconnect(): Promise<void> {
    if (this.channel) {
      await this.channel.close();
      this.channel = null;
    }
    if (this.connection) {
      await this.connection.close();
      this.connection = null;
    }
    logger.info('RabbitMQ disconnected');
  }

  getChannel(): Channel | null {
    return this.channel;
  }
}

export default RabbitMQClient;
