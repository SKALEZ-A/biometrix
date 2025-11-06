import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { CacheManager } from '../../src/utils/cache.utils';
import Redis from 'ioredis-mock';

describe('CacheManager', () => {
  let redis: Redis;
  let cacheManager: CacheManager;

  beforeEach(() => {
    redis = new Redis();
    cacheManager = new CacheManager(redis, { prefix: 'test:' });
  });

  afterEach(async () => {
    await redis.flushall();
    redis.disconnect();
  });

  describe('get and set', () => {
    it('should store and retrieve string values', async () => {
      await cacheManager.set('key1', 'value1');
      const value = await cacheManager.get<string>('key1');

      expect(value).toBe('value1');
    });

    it('should store and retrieve object values', async () => {
      const obj = { name: 'John', age: 30 };
      await cacheManager.set('key2', obj);
      const value = await cacheManager.get<typeof obj>('key2');

      expect(value).toEqual(obj);
    });

    it('should return null for non-existent keys', async () => {
      const value = await cacheManager.get('non-existent');

      expect(value).toBeNull();
    });

    it('should respect TTL', async () => {
      await cacheManager.set('key3', 'value3', 1);
      
      let value = await cacheManager.get('key3');
      expect(value).toBe('value3');

      await new Promise(resolve => setTimeout(resolve, 1100));

      value = await cacheManager.get('key3');
      expect(value).toBeNull();
    });
  });

  describe('delete', () => {
    it('should delete existing keys', async () => {
      await cacheManager.set('key4', 'value4');
      await cacheManager.delete('key4');

      const value = await cacheManager.get('key4');
      expect(value).toBeNull();
    });
  });

  describe('getOrSet', () => {
    it('should return cached value if exists', async () => {
      await cacheManager.set('key5', 'cached-value');

      const value = await cacheManager.getOrSet('key5', async () => 'new-value');

      expect(value).toBe('cached-value');
    });

    it('should call factory and cache result if not exists', async () => {
      const factory = jest.fn(async () => 'factory-value');

      const value = await cacheManager.getOrSet('key6', factory);

      expect(value).toBe('factory-value');
      expect(factory).toHaveBeenCalledTimes(1);

      const cachedValue = await cacheManager.get('key6');
      expect(cachedValue).toBe('factory-value');
    });
  });
});
