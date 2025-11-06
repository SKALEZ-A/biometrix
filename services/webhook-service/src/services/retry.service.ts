import { logger } from '@shared/utils/logger';

export interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors?: string[];
}

export interface RetryJob {
  jobId: string;
  attempt: number;
  maxAttempts: number;
  nextRetryAt: Date;
  data: any;
  error?: string;
}

export class RetryService {
  private retryQueue: Map<string, RetryJob> = new Map();
  private defaultConfig: RetryConfig = {
    maxRetries: 5,
    initialDelay: 1000,
    maxDelay: 60000,
    backoffMultiplier: 2,
    retryableErrors: ['ECONNREFUSED', 'ETIMEDOUT', 'ENOTFOUND'],
  };

  async executeWithRetry<T>(
    fn: () => Promise<T>,
    config: Partial<RetryConfig> = {}
  ): Promise<T> {
    const finalConfig = { ...this.defaultConfig, ...config };
    let lastError: Error | undefined;
    
    for (let attempt = 1; attempt <= finalConfig.maxRetries; attempt++) {
      try {
        logger.info(`Executing function (attempt ${attempt}/${finalConfig.maxRetries})`);
        return await fn();
      } catch (error: any) {
        lastError = error;
        
        // Check if error is retryable
        if (!this.isRetryableError(error, finalConfig)) {
          logger.error('Non-retryable error encountered:', error);
          throw error;
        }
        
        if (attempt < finalConfig.maxRetries) {
          const delay = this.calculateDelay(attempt, finalConfig);
          logger.warn(`Attempt ${attempt} failed, retrying in ${delay}ms:`, error.message);
          await this.sleep(delay);
        }
      }
    }
    
    logger.error(`All ${finalConfig.maxRetries} attempts failed`);
    throw lastError;
  }

  private isRetryableError(error: any, config: RetryConfig): boolean {
    if (!config.retryableErrors || config.retryableErrors.length === 0) {
      return true; // Retry all errors if no specific errors defined
    }
    
    const errorCode = error.code || error.message;
    return config.retryableErrors.some(retryableError =>
      errorCode.includes(retryableError)
    );
  }

  private calculateDelay(attempt: number, config: RetryConfig): number {
    const delay = config.initialDelay * Math.pow(config.backoffMultiplier, attempt - 1);
    return Math.min(delay, config.maxDelay);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async scheduleRetry(jobId: string, data: any, config: Partial<RetryConfig> = {}): Promise<void> {
    const finalConfig = { ...this.defaultConfig, ...config };
    
    const job: RetryJob = {
      jobId,
      attempt: 1,
      maxAttempts: finalConfig.maxRetries,
      nextRetryAt: new Date(Date.now() + finalConfig.initialDelay),
      data,
    };
    
    this.retryQueue.set(jobId, job);
    logger.info(`Retry job scheduled: ${jobId}`);
  }

  async processRetryQueue(): Promise<void> {
    const now = new Date();
    
    for (const [jobId, job] of this.retryQueue.entries()) {
      if (job.nextRetryAt <= now) {
        try {
          await this.processRetryJob(job);
          this.retryQueue.delete(jobId);
        } catch (error: any) {
          if (job.attempt >= job.maxAttempts) {
            logger.error(`Job ${jobId} failed after ${job.maxAttempts} attempts`);
            this.retryQueue.delete(jobId);
          } else {
            job.attempt++;
            job.nextRetryAt = new Date(Date.now() + this.calculateDelay(job.attempt, this.defaultConfig));
            job.error = error.message;
            this.retryQueue.set(jobId, job);
          }
        }
      }
    }
  }

  private async processRetryJob(job: RetryJob): Promise<void> {
    logger.info(`Processing retry job: ${job.jobId} (attempt ${job.attempt}/${job.maxAttempts})`);
    // Implementation specific to job type
  }

  getRetryQueueSize(): number {
    return this.retryQueue.size;
  }

  getRetryJob(jobId: string): RetryJob | undefined {
    return this.retryQueue.get(jobId);
  }

  cancelRetryJob(jobId: string): boolean {
    return this.retryQueue.delete(jobId);
  }

  clearRetryQueue(): void {
    this.retryQueue.clear();
    logger.info('Retry queue cleared');
  }
}
