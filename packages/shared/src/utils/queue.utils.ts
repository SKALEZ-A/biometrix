export interface QueueJob<T = any> {
  id: string;
  data: T;
  priority: number;
  attempts: number;
  maxAttempts: number;
  createdAt: Date;
  processedAt?: Date;
  error?: string;
}

export class InMemoryQueue<T = any> {
  private queue: QueueJob<T>[] = [];
  private processing: boolean = false;
  private processor?: (job: QueueJob<T>) => Promise<void>;

  constructor(private concurrency: number = 1) {}

  async add(data: T, priority: number = 0, maxAttempts: number = 3): Promise<string> {
    const job: QueueJob<T> = {
      id: this.generateId(),
      data,
      priority,
      attempts: 0,
      maxAttempts,
      createdAt: new Date()
    };

    this.queue.push(job);
    this.queue.sort((a, b) => b.priority - a.priority);

    if (!this.processing) {
      this.process();
    }

    return job.id;
  }

  setProcessor(processor: (job: QueueJob<T>) => Promise<void>): void {
    this.processor = processor;
  }

  private async process(): Promise<void> {
    if (this.processing || !this.processor) return;

    this.processing = true;

    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.concurrency);

      await Promise.all(
        batch.map(async (job) => {
          try {
            job.attempts++;
            await this.processor!(job);
            job.processedAt = new Date();
          } catch (error) {
            job.error = error instanceof Error ? error.message : String(error);

            if (job.attempts < job.maxAttempts) {
              this.queue.push(job);
            }
          }
        })
      );
    }

    this.processing = false;
  }

  getQueueSize(): number {
    return this.queue.length;
  }

  clear(): void {
    this.queue = [];
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
