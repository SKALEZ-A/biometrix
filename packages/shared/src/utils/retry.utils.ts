export interface RetryConfig {
  maxAttempts: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors?: string[];
}

export class RetryError extends Error {
  constructor(
    message: string,
    public attempts: number,
    public lastError: Error
  ) {
    super(message);
    this.name = 'RetryError';
  }
}

export async function retry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    backoffMultiplier = 2,
    retryableErrors = []
  } = config;

  let lastError: Error;
  let delay = initialDelay;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      // Check if error is retryable
      if (retryableErrors.length > 0) {
        const isRetryable = retryableErrors.some(errType =>
          lastError.name === errType || lastError.message.includes(errType)
        );
        if (!isRetryable) {
          throw lastError;
        }
      }

      // Don't wait after last attempt
      if (attempt === maxAttempts) {
        break;
      }

      // Wait before next attempt
      await sleep(delay);
      delay = Math.min(delay * backoffMultiplier, maxDelay);
    }
  }

  throw new RetryError(
    `Failed after ${maxAttempts} attempts`,
    maxAttempts,
    lastError!
  );
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function retryWithJitter<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    backoffMultiplier = 2
  } = config;

  let lastError: Error;
  let delay = initialDelay;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === maxAttempts) {
        break;
      }

      // Add jitter to prevent thundering herd
      const jitter = Math.random() * delay * 0.3;
      const actualDelay = Math.min(delay + jitter, maxDelay);

      await sleep(actualDelay);
      delay = delay * backoffMultiplier;
    }
  }

  throw new RetryError(
    `Failed after ${maxAttempts} attempts`,
    maxAttempts,
    lastError!
  );
}
