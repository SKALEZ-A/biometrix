import { describe, it, expect, jest } from '@jest/globals';
import { retry, RetryError } from '../../src/utils/retry.utils';

describe('retry', () => {
  it('should succeed on first attempt', async () => {
    const fn = jest.fn(async () => 'success');

    const result = await retry(fn, { maxAttempts: 3 });

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should retry on failure and eventually succeed', async () => {
    let attempts = 0;
    const fn = jest.fn(async () => {
      attempts++;
      if (attempts < 3) {
        throw new Error('Temporary failure');
      }
      return 'success';
    });

    const result = await retry(fn, { maxAttempts: 5, initialDelay: 10 });

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should throw RetryError after max attempts', async () => {
    const fn = jest.fn(async () => {
      throw new Error('Permanent failure');
    });

    await expect(
      retry(fn, { maxAttempts: 3, initialDelay: 10 })
    ).rejects.toThrow(RetryError);

    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should respect retryable errors', async () => {
    const fn = jest.fn(async () => {
      const error = new Error('Network error');
      error.name = 'NetworkError';
      throw error;
    });

    await expect(
      retry(fn, {
        maxAttempts: 3,
        initialDelay: 10,
        retryableErrors: ['NetworkError']
      })
    ).rejects.toThrow(RetryError);

    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should not retry non-retryable errors', async () => {
    const fn = jest.fn(async () => {
      const error = new Error('Validation error');
      error.name = 'ValidationError';
      throw error;
    });

    await expect(
      retry(fn, {
        maxAttempts: 3,
        initialDelay: 10,
        retryableErrors: ['NetworkError']
      })
    ).rejects.toThrow('Validation error');

    expect(fn).toHaveBeenCalledTimes(1);
  });
});
